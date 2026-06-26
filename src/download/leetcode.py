"""
LeetCode problem downloader using Playwright.
Downloads LeetCode problems from GitHub Codespaces LeetCode plugin and saves them as markdown files.
"""
from __future__ import annotations
import asyncio
import os
import json
import re
import time
from typing import Optional, Any, List, Dict, TYPE_CHECKING
from markdownify import markdownify as md
if TYPE_CHECKING:  # playwright is an optional extra (only needed to run the leetcode downloader)
    from playwright.async_api import Playwright, BrowserContext, Page
from dotenv import load_dotenv

load_dotenv(verbose=True)

from src.download.type import AbstractDownloader
from src.logger import logger
from src.utils import assemble_project_path


class LeetCodeDownloader(AbstractDownloader):
    """Downloader for LeetCode problems using Playwright automation via GitHub Codespaces."""
    
    def __init__(self,
                 start_id: int = 1,
                 end_id: int = 10,
                 output_dir: Optional[str] = None,
                 output_jsonl: Optional[str] = None,
                 github_username: Optional[str] = None,
                 github_password: Optional[str] = None,
                 project_url: Optional[str] = None,
                 leetcode_cookie: Optional[str] = None,
                 codespace_name: str = "fuzzy memory",
                 headless: bool = True,
                 max_scroll_attempts: int = 20,
                 **kwargs):
        super().__init__()
        
        self.start_id = start_id
        self.end_id = end_id
        self.output_dir = assemble_project_path(output_dir) if output_dir else assemble_project_path("leetcode_problems")
        self.output_jsonl = output_jsonl if output_jsonl else os.path.join(self.output_dir, "leetcode_index.jsonl")
        self.github_username = github_username or os.getenv("GITHUB_USERNAME")
        self.github_password = github_password or os.getenv("GITHUB_PASSWORD")
        self.project_url = project_url or os.getenv("PROJECT_URL")
        self.leetcode_cookie = leetcode_cookie or os.getenv("LEETCODE_COOKIE", "")
        self.codespace_name = codespace_name
        self.headless = headless
        self.max_scroll_attempts = max_scroll_attempts
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Browser context
        self.browser_context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
    def _sanitize_filename(self, text: str) -> str:
        """
        Sanitize filename:
        1. Convert to lowercase
        2. Replace spaces with underscores
        3. Remove illegal characters
        Example: "Two Sum" -> "two_sum"
        """
        text = text.lower()
        text = text.replace(" ", "_")
        text = re.sub(r'[^\w\u4e00-\u9fa5\._-]', '', text)
        return text
    
    async def _get_max_visible_id(self, page: Page) -> int:
        """
        Get the maximum visible problem ID in the current viewport.
        """
        try:
            visible_items = await page.locator('[role="treeitem"]').all_text_contents()
            max_id = -1
            pattern = re.compile(r'\[(\d+)\]')
            
            for item_text in visible_items:
                match = pattern.search(item_text)
                if match:
                    current_id = int(match.group(1))
                    if current_id > max_id:
                        max_id = current_id
            
            return max_id
        except Exception as e:
            logger.error(f"Failed to get max visible ID: {e}")
            return -1
    
    async def _close_all_editors(self, page: Page):
        """
        Close all editors to prevent tab accumulation.
        """
        try:
            await page.mouse.click(100, 100)
            await page.keyboard.press("F1")
            await asyncio.sleep(0.5)
            await page.keyboard.type("View: Close All Editors")
            await asyncio.sleep(0.5)
            await page.keyboard.press("Enter")
            await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"Failed to close editors: {e}")
    
    async def _load_existing_ids(self) -> set:
        """
        Load existing problem IDs from JSONL file.
        """
        existing_ids = set()
        if os.path.exists(self.output_jsonl):
            try:
                with open(self.output_jsonl, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if "id" in data:
                                existing_ids.add(int(data["id"]))
                        except:
                            continue
            except Exception as e:
                logger.warning(f"Failed to load existing IDs: {e}")
        else:
            # Create empty file if it doesn't exist
            with open(self.output_jsonl, 'w', encoding='utf-8') as f:
                pass
        return existing_ids
    
    async def _find_problem_item(self, page: Page, problem_id: int) -> Optional[Any]:
        """
        Find and return the problem item locator for the given problem ID.
        Returns None if not found.
        """
        problem_text_pattern = re.compile(f"\\[{problem_id}\\]\\s")
        
        for attempt in range(self.max_scroll_attempts):
            # Try to locate the problem
            locator = page.locator(f'[role="treeitem"]').filter(has_text=problem_text_pattern).first
            
            if await locator.count() > 0 and await locator.is_visible():
                logger.info(f"✅ Found problem [{problem_id}]")
                return locator
            
            # Check if we've scrolled past the problem
            current_max_id = await self._get_max_visible_id(page)
            logger.debug(f"Current screen max ID: {current_max_id}, target ID: {problem_id}")
            
            if current_max_id > problem_id:
                # If we've scrolled past the problem, it doesn't exist
                logger.warning(f"⚠️ Problem [{problem_id}] does not exist (current max: {current_max_id})")
                return None
            
            # Scroll down if we haven't reached the problem yet
            if current_max_id < problem_id:
                logger.debug(f"Scrolling down (attempt {attempt+1})...")
                tree_container = page.locator('div[role="tree"]').first
                if await tree_container.is_visible():
                    await tree_container.hover()
                    await page.mouse.wheel(0, 600)
                    await asyncio.sleep(1.0)
                else:
                    break
        
        return None
    
    async def _extract_problem_content(self, page: Page, problem_name: str) -> str:
        """
        Extract problem content from the webview frame.
        Returns "PREMIUM_LOCKED" if the problem is premium-locked.
        """
        html_content = ""
        deadline = time.time() + 10
        
        # Normalize function: lowercase -> unify quotes -> replace whitespace with single space
        normalize = lambda s: " ".join(s.lower().replace("'", "'").replace("'", "'").split())
        target_name_clean = normalize(problem_name)
        
        while time.time() < deadline:
            for frame in page.frames:
                try:
                    body_text = await frame.locator('body').inner_text()
                    
                    if "Subscribe to unlock" in body_text:
                        return "PREMIUM_LOCKED"
                    
                    # Check for problem content indicators
                    if (await frame.locator('text="Example 1:"').count() > 0 or
                        await frame.locator('text="Description"').count() > 0 or
                        await frame.locator('h1').count() > 0):
                        
                        page_content_clean = normalize(body_text)
                        
                        # Fuzzy match
                        if target_name_clean in page_content_clean:
                            html_content = await frame.locator('body').inner_html()
                            logger.info(f"✅ Content matched successfully")
                            break
                except:
                    continue
            
            if html_content:
                break
            
            logger.debug(".", end="", flush=True)
            await asyncio.sleep(1)
        
        return html_content
    
    async def _download_problem(self, page: Page, problem_id: int) -> bool:
        """
        Download a single LeetCode problem.
        Returns True if successful, False otherwise.
        """
        try:
            await self._close_all_editors(page)
            
            # Find the problem item
            problem_item = await self._find_problem_item(page, problem_id)
            if not problem_item:
                logger.warning(f"⏭️ Problem [{problem_id}] not found, skipping")
                return False
            
            # Extract problem name
            await problem_item.scroll_into_view_if_needed()
            full_text = await problem_item.text_content()
            logger.info(f"Target problem: {full_text}")
            
            problem_name_raw = "unknown"
            if f"[{problem_id}]" in full_text:
                parts = full_text.split(f"[{problem_id}]")
                if len(parts) > 1:
                    problem_name_raw = parts[1].strip().replace("🔓", "").replace("🔒", "").strip()
            
            # Click the problem
            await problem_item.click()
            
            # Wait for content and extract
            logger.info("Waiting for webview to update...")
            html_content = await self._extract_problem_content(page, problem_name_raw)
            
            if not html_content or html_content == "PREMIUM_LOCKED":
                logger.warning(f"❌ Skipping problem [{problem_id}] (premium or failed to fetch)")
                await page.keyboard.press("Control+W")
                return False
            
            # Convert to markdown
            markdown_content = md(html_content, heading_style="ATX")
            markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)
            header = f"# {problem_id}. {problem_name_raw}\n\n"
            final_md = header + markdown_content
            
            # Save to file
            safe_name = self._sanitize_filename(problem_name_raw)
            file_name = f"{problem_id}.{safe_name}.md"
            file_path = os.path.join(self.output_dir, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(final_md)
            
            # Update index
            relative_path = f"./{os.path.basename(self.output_dir)}/{file_name}"
            index_data = {
                "id": problem_id,
                "name": problem_name_raw,
                "file": relative_path
            }
            
            with open(self.output_jsonl, 'a', encoding='utf-8') as f:
                f.write(json.dumps(index_data, ensure_ascii=False) + "\n")
            
            logger.info(f"💾 Problem [{problem_id}] saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error downloading problem [{problem_id}]: {e}")
            try:
                await page.keyboard.press("Control+W")
            except:
                pass
            return False
    
    async def _setup_browser(self, playwright: Playwright):
        """
        Setup browser with persistent context.
        """
        user_data_dir = os.path.join(os.getcwd(), "playwright_user_data")
        self.browser_context = await playwright.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=self.headless,
            viewport={'width': 1280, 'height': 800},
            args=["--disable-blink-features=AutomationControlled"]
        )
        
        if len(self.browser_context.pages) > 0:
            self.page = self.browser_context.pages[0]
        else:
            self.page = await self.browser_context.new_page()
        
        logger.info(f"Browser started, data saved to: {user_data_dir}")
    
    async def _navigate_to_github(self):
        """Navigate to GitHub."""
        logger.info("Navigating to GitHub...")
        await self.page.goto("https://github.com")
        await asyncio.sleep(2)
    
    async def _click_sign_in(self):
        """Click sign in button."""
        try:
            sign_in_button = self.page.locator('a:text("Sign in")').first
            await sign_in_button.click(timeout=5000)
        except:
            await self.page.goto("https://github.com/login")
    
    async def _login_github(self):
        """Login to GitHub."""
        logger.info("Logging in to GitHub...")
        try:
            if await self.page.locator('input[name="login"]').count() == 0:
                logger.info("Already logged in or login form not found")
                return
            
            await self.page.locator('input[name="login"]').fill(self.github_username)
            await self.page.locator('input[name="password"]').fill(self.github_password)
            await self.page.locator('input[type="submit"][value="Sign in"]').click()
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Login error: {e}")
    
    async def _navigate_to_project(self):
        """Navigate to GitHub project."""
        logger.info(f"Navigating to project: {self.project_url}")
        await self.page.goto(self.project_url)
        await asyncio.sleep(3)
    
    async def _click_code_button(self):
        """Click Code button."""
        try:
            code_button = self.page.locator('button[data-variant="primary"]:has-text("Code")').first
            await code_button.click()
            await asyncio.sleep(2)
        except Exception as e:
            logger.warning(f"Failed to click Code button: {e}")
    
    async def _enter_codespaces(self):
        """Enter Codespaces."""
        logger.info("Entering Codespaces...")
        try:
            await self.page.locator('[role="tab"]:has-text("Codespaces")').click()
            await asyncio.sleep(2)
            await self.page.get_by_text(self.codespace_name, exact=False).first.click()
            logger.info(f"Waiting for Codespace to start (25 seconds)...")
            await asyncio.sleep(25)
        except Exception as e:
            logger.error(f"Error entering Codespaces: {e}")
    
    async def _open_leetcode_plugin(self):
        """Open LeetCode plugin."""
        logger.info("Opening LeetCode plugin...")
        try:
            await self.page.locator('a[aria-label="LeetCode"]').first.click()
            await asyncio.sleep(3)
        except Exception as e:
            logger.warning(f"Failed to open LeetCode plugin: {e}")
    
    async def _login_leetcode(self):
        """Login to LeetCode (if needed)."""
        logger.info("Logging in to LeetCode...")
        # Cookie-based login is handled via browser context
        # Additional login logic can be added here if needed
        pass
    
    async def _click_all_problems(self):
        """Click the 'All' button to expand the problem list."""
        logger.info("Clicking All button to expand problem list...")
        try:
            all_btn = self.page.locator('div[role="treeitem"] >> text="All"').first
            if await all_btn.is_visible():
                await all_btn.click()
            else:
                await self.page.locator('span:has-text("All")').first.click()
            
            logger.info("Clicked All button")
            await asyncio.sleep(3)
        except Exception as e:
            logger.warning(f"Failed to click All button (may already be expanded): {e}")
    
    async def run(self):
        """
        Main download method.
        """
        if not self.github_username or not self.github_password or not self.project_url:
            raise ValueError("GitHub username, password, and project_url are required")
        
        existing_ids = await self._load_existing_ids()
        success_count = 0
        
        from playwright.async_api import async_playwright  # optional dependency
        async with async_playwright() as playwright:
            try:
                # 1. Setup browser
                await self._setup_browser(playwright)
                
                # 2. Navigate to GitHub and login
                await self._navigate_to_github()
                await self._click_sign_in()
                await self._login_github()
                
                # 3. Navigate to project and enter Codespaces
                await self._navigate_to_project()
                await self._click_code_button()
                await self._enter_codespaces()
                
                # 4. Open LeetCode plugin
                await self._open_leetcode_plugin()
                await self._login_leetcode()
                
                # 5. Click All problems button
                await self._click_all_problems()
                
                # 6. Download problems
                logger.info(f"\nStarting to download problems [{self.start_id}] to [{self.end_id}]...")
                for problem_id in range(self.start_id, self.end_id + 1):
                    if problem_id in existing_ids:
                        logger.info(f"⏩ Problem [{problem_id}] already exists, skipping")
                        continue
                    
                    logger.info(f"\n--- Processing problem [{problem_id}] ---")
                    success = await self._download_problem(self.page, problem_id)
                    if success:
                        success_count += 1
                    
                    # Small delay between problems
                    await asyncio.sleep(1)
                
                logger.info(f"\n✅ Download complete! Successfully downloaded {success_count} new problems.")
                
            except Exception as e:
                logger.error(f"Error during download: {e}")
                if self.page:
                    await self.page.screenshot(path=os.path.join(self.output_dir, "error.png"))
                raise
            finally:
                if self.browser_context:
                    await self.browser_context.close()
                logger.info("Browser closed")
