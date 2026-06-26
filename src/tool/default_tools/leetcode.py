"""LeetCode tool for fetching problem information from LeetCode."""

import aiohttp
from typing import Optional, Dict, Any, ClassVar
from pydantic import Field

from src.logger import logger
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.registry import TOOL

_LEETCODE_DESCRIPTION = """Fetch LeetCode problem information by problem slug or ID.
Use this tool to get problem details including title, content, difficulty, tags, and more.
You can search by problem slug (e.g., 'two-sum') or by problem ID.
"""

@TOOL.register_module(force=True)
class LeetCodeTool(Tool):
    """A tool for fetching LeetCode problem information asynchronously."""

    name: str = "leetcode"
    description: str = _LEETCODE_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    
    # LeetCode GraphQL endpoint
    GRAPHQL_URL: ClassVar[str] = "https://leetcode.com/graphql"
    API_URL: ClassVar[str] = "https://leetcode.com/api/problems/all/"
    
    def __init__(self, **kwargs):
        """Initialize the LeetCode tool."""
        super().__init__(**kwargs)

    async def __call__(self, slug: Optional[str] = None, problem_id: Optional[int] = None, **kwargs) -> ToolResponse:
        """
        Fetch LeetCode problem information by slug or problem ID.

        Args:
            slug (str, optional): The problem slug (e.g., 'two-sum'). Defaults to None.
            problem_id (int, optional): The problem ID (e.g., 1). Defaults to None.
        
        Returns:
            ToolResponse: The response containing problem information.
        """
        try:
            if not slug and not problem_id:
                return ToolResponse(
                    success=False,
                    message="Either 'slug' or 'problem_id' must be provided",
                )
            
            # If only problem_id is provided, we need to get the slug first
            if problem_id and not slug:
                slug = await self._get_slug_by_id(problem_id)
                if not slug:
                    return ToolResponse(
                        success=False,
                        message=f"Problem with ID {problem_id} not found"
                    )
            
            # Fetch problem details using GraphQL
            problem_data = await self._fetch_problem_details(slug)
            
            if not problem_data:
                return ToolResponse(
                    success=False,
                    message=f"Failed to fetch problem details for slug: {slug}"
                )
            
            # Format the response
            formatted_response = self._format_problem_response(problem_data)
            
            return ToolResponse(
                success=True,
                message=formatted_response,
                extra=ToolExtra(
                    file_path=None,
                    data={
                        "slug": slug,
                        "problem_id": problem_id,
                        "problem_data": problem_data
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"Error fetching LeetCode problem: {e}")
            return ToolResponse(
                success=False,
                message=f"Error fetching LeetCode problem: {str(e)}"
            )

    async def _get_slug_by_id(self, problem_id: int) -> Optional[str]:
        """Get problem slug by problem ID."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                async with session.get(self.API_URL, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        questions = data.get('stat_status_pairs', [])
                        for question in questions:
                            if question['stat']['question_id'] == problem_id:
                                return question['stat']['question__title_slug']
                    return None
        except Exception as e:
            logger.error(f"Error getting slug by ID: {e}")
            return None

    async def _fetch_problem_details(self, slug: str) -> Optional[Dict[str, Any]]:
        """Fetch problem details using LeetCode GraphQL API."""
        query = """
        query getQuestionDetail($titleSlug: String!) {
            question(titleSlug: $titleSlug) {
                questionId
                questionFrontendId
                title
                titleSlug
                content
                difficulty
                likes
                dislikes
                isLiked
                similarQuestions
                contributors {
                    username
                    profileUrl
                    avatarUrl
                }
                topicTags {
                    name
                    slug
                    translatedName
                }
                codeSnippets {
                    lang
                    langSlug
                    code
                }
                stats
                hints
                solution {
                    id
                    canSeeDetail
                    paidOnly
                    hasVideoSolution
                    paidOnlyVideo
                }
                status
                sampleTestCase
                exampleTestcases
                metaData
                judgerAvailable
                judgeType
                mysqlSchemas
                enableRunCode
                enableTestMode
                enableDebugger
                envInfo
                libraryUrl
                adminUrl
                challengeQuestion {
                    id
                    date
                    incompleteChallengeCount
                    streakCount
                    type
                }
                note
            }
        }
        """
        
        variables = {"titleSlug": slug}
        
        payload = {
            "operationName": "getQuestionDetail",
            "variables": variables,
            "query": query
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                async with session.post(self.GRAPHQL_URL, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and 'question' in data['data']:
                            return data['data']['question']
                    logger.warning(f"Failed to fetch problem details. Status: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching problem details: {e}")
            return None

    def _format_problem_response(self, problem_data: Dict[str, Any]) -> str:
        """Format problem data into a readable string."""
        lines = []
        
        # Basic information
        lines.append(f"# {problem_data.get('title', 'Unknown')}")
        lines.append(f"**Problem ID:** {problem_data.get('questionFrontendId', 'N/A')}")
        lines.append(f"**Difficulty:** {problem_data.get('difficulty', 'Unknown')}")
        lines.append("")
        
        # Content
        content = problem_data.get('content', '')
        if content:
            lines.append("## Problem Description")
            lines.append(content)
            lines.append("")
        
        # Tags
        tags = problem_data.get('topicTags', [])
        if tags:
            tag_names = [tag.get('name', '') for tag in tags]
            lines.append(f"**Tags:** {', '.join(tag_names)}")
            lines.append("")
        
        # Code snippets
        code_snippets = problem_data.get('codeSnippets', [])
        if code_snippets:
            lines.append("## Code Snippets")
            for snippet in code_snippets:
                lang = snippet.get('lang', 'Unknown')
                code = snippet.get('code', '')
                lines.append(f"### {lang}")
                lines.append(f"```{snippet.get('langSlug', '')}")
                lines.append(code)
                lines.append("```")
                lines.append("")
        
        # Hints
        hints = problem_data.get('hints', [])
        if hints:
            lines.append("## Hints")
            for i, hint in enumerate(hints, 1):
                lines.append(f"{i}. {hint}")
            lines.append("")
        
        # Stats
        stats = problem_data.get('stats', '')
        if stats:
            import json
            try:
                stats_data = json.loads(stats)
                lines.append("## Statistics")
                lines.append(f"- **Total Accepted:** {stats_data.get('totalAccepted', 'N/A')}")
                lines.append(f"- **Total Submissions:** {stats_data.get('totalSubmission', 'N/A')}")
                lines.append(f"- **Acceptance Rate:** {stats_data.get('acRate', 'N/A')}")
                lines.append("")
            except:
                pass
        
        return "\n".join(lines)

