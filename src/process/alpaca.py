import os
import pandas as pd
import asyncio
from typing import Optional, Any
from datetime import datetime
from markdownify import markdownify

from src.process.base import AbstractProcessor
from src.registry import PROCESSOR
from src.model import model_manager
from src.message import HumanMessage
from src.logger import logger
from src.factor import factor_manager
from src.config import config
from src.utils import gather_with_concurrency

class AlpacaPriceProcessor(AbstractProcessor):
    def __init__(self,
                 data_path: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 symbol_info: Optional[Any] = None,
                 workdir: Optional[str] = None,
                 ):
        super().__init__()

        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format
        self.max_concurrent = max_concurrent

        self.symbol_info = symbol_info
        self.symbol = symbol_info["symbol"] if symbol_info else None

        self.workdir = workdir

    async def run(self):
        info = {
            "symbol": self.symbol,
        }

        price_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        price_column_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

        data_path = os.path.join(self.data_path, "{}.jsonl".format(self.symbol))

        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")

        assert os.path.exists(data_path), "Price path {} does not exist".format(data_path)

        df = pd.read_json(data_path, lines=True)

        df = df.rename(columns=price_column_map)[["timestamp"] + price_columns]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        df = df.sort_values(by="timestamp")
        df = df[(df["timestamp"] >= start_date) & (df["timestamp"] < end_date)]
        df = df.reset_index(drop=True)
        df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

        info["names"] = list(sorted(df.columns))
        info["start_date"] = df["timestamp"].min()
        info["end_date"] = df["timestamp"].max()

        df.to_json(os.path.join(self.workdir, "{}.jsonl".format(self.symbol)), orient="records", lines=True)

        return info


class AlpacaFeatureProcessor(AbstractProcessor):
    def __init__(self,
                 data_path: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 symbol_info: Optional[Any] = None,
                 workdir: Optional[str] = None,
                 ):
        super().__init__()

        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format
        self.max_concurrent = max_concurrent

        self.symbol_info = symbol_info
        self.symbol = symbol_info["symbol"] if symbol_info else None

        self.workdir = workdir

    async def run(self):
        info = {
            "symbol": self.symbol,
        }

        price_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        price_column_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

        data_path = os.path.join(self.data_path, "{}.jsonl".format(self.symbol))

        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")

        assert os.path.exists(data_path), "Price path {} does not exist".format(data_path)

        df = pd.read_json(data_path, lines=True)

        df = df.rename(columns=price_column_map)[["timestamp"] + price_columns]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        df = df.sort_values(by="timestamp")
        df = df[(df["timestamp"] >= start_date) & (df["timestamp"] < end_date)]
        df = df.reset_index(drop=True)
        df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

        factors_df = await gather_with_concurrency(
            [factor_manager(factor_name, df) for factor_name in config.factors],
            max_concurrency=self.max_concurrent,
        )
        factors_df = pd.concat(factors_df, axis=1)

        info["names"] = list(sorted(factors_df.columns))
        info["start_date"] = df["timestamp"].min()
        info["end_date"] = df["timestamp"].max()

        factors_df.to_json(os.path.join(self.workdir, "{}.jsonl".format(self.symbol)), orient="records", lines=True)

        return info


class AlpacaNewsProcessor(AbstractProcessor):
    def __init__(self,
                 data_path: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 symbol_info: Optional[Any] = None,
                 workdir: Optional[str] = None,
                 ):
        super().__init__()

        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format
        self.max_concurrent = max_concurrent
        self.symbol_info = symbol_info
        self.symbol = symbol_info["symbol"] if symbol_info else None
        self.workdir = workdir
        
    async def _summary(self, df: pd.DataFrame):
        """
        Use LLM to perform async parallel summarization of financial news content
        
        Args:
            df: DataFrame containing news data, must include 'content' column
            
        Returns:
            pd.DataFrame: DataFrame with added 'summary' column
        """
        if df.empty or 'content' not in df.columns:
            logger.warning("DataFrame is empty or doesn't contain content column, skipping summarization")
            return df
            
        # Ensure model_manager is initialized
        if not model_manager.registed_models:
            await model_manager.initialize()
            
        # Select model for summarization
        model_name = "gpt-5-mini"  # Can be adjusted as needed
        if model_name not in model_manager.registed_models:
            logger.warning(f"Model {model_name} not registered, using default model")
            model_name = list(model_manager.registed_models.keys())[0]
            
        model = model_manager.registed_models[model_name]
        
        # Design prompt for financial news summarization
        summary_prompt = """Please provide a comprehensive yet concise summary of the following financial news. 
        Your summary should capture ALL critical information without losing important details:

        Key elements to include:
        - Specific company names, stock symbols, and financial figures
        - Exact dates, percentages, dollar amounts, and metrics
        - Market reactions, analyst opinions, and regulatory actions
        - Strategic decisions, partnerships, acquisitions, or divestitures
        - Earnings data, revenue numbers, and growth rates
        - Risk factors, challenges, and future outlook

        Requirements:
        - Be precise and factual - preserve all numerical data and specific details
        - Maintain chronological order when relevant
        - Include both positive and negative aspects
        - Keep the summary comprehensive but well-structured
        - Use clear, professional language
        - Aim for 150-200 words to ensure no important information is lost

        Financial news content:
        {content}"""
        
        async def summarize_single_news(content: str, index: int) -> tuple[int, str]:
            """Summarize single news content"""
            try:
                if not content or not content.strip():
                    return index, ""
                    
                # Limit content length to avoid token overflow
                max_content_length = 10000
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                
                prompt = summary_prompt.format(content=content)
                messages = [HumanMessage(content=prompt)]
                
                response = await model.ainvoke(messages)
                summary = response.content.strip()
                
                logger.debug(f"Successfully summarized news {index}, length: {len(summary)}")
                return index, summary
                
            except Exception as e:
                logger.error(f"Error summarizing news {index}: {str(e)}")
                return index, f"Summarization failed: {str(e)[:50]}"
        
        # Create summarization task list
        tasks = []
        for idx, row in df.iterrows():
            content = row.get('content', '')
            task = summarize_single_news(content, idx)
            tasks.append(task)
        
        # Use semaphore to control concurrency and avoid API limits
        max_concurrent = min(self.max_concurrent or 10, len(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_summarize(content: str, index: int) -> tuple[int, str]:
            async with semaphore:
                return await summarize_single_news(content, index)
        
        # Recreate tasks with concurrency limits
        limited_tasks = [
            limited_summarize(row.get('content', ''), idx) 
            for idx, row in df.iterrows()
        ]
        
        logger.info(f"Starting parallel summarization of {len(limited_tasks)} news items, max concurrency: {max_concurrent}")
        
        try:
            # Use asyncio.gather for parallel processing
            results = await asyncio.gather(*limited_tasks, return_exceptions=True)
            
            # Process results
            summaries = [""] * len(df)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Summarization task exception: {str(result)}")
                    continue
                    
                index, summary = result
                if 0 <= index < len(summaries):
                    summaries[index] = summary
            
            # Add summary column to DataFrame
            df = df.copy()
            df['summary'] = summaries
            
            logger.info(f"Completed summarization of {len(df)} news items, successful: {sum(1 for s in summaries if s and not s.startswith('Summarization failed'))}")
            
        except Exception as e:
            logger.error(f"Error during parallel summarization: {str(e)}")
        
        return df

    async def run(self):

        info = {
            "symbol": self.symbol,
        }

        news_columns = [
            "title",
            "content",
            "raw_content"
        ]
        news_column_map = {
            "headline": "title",
            "content": "content",
            "timestamp": "timestamp",
        }

        data_path = os.path.join(self.data_path, "{}.jsonl".format(self.symbol))

        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")

        assert os.path.exists(data_path), "News path {} does not exist".format(data_path)

        df = pd.read_json(data_path, lines=True)
        df = df.rename(columns=news_column_map)[["timestamp"] + news_columns]

        def process_content(row):
            content = row['content']
            raw_content = row['raw_content']

            if isinstance(content, str) and content.strip():
                return content
            elif isinstance(raw_content, str):
                return markdownify(raw_content)
            else:
                return ""

        df['content'] = df.apply(process_content, axis=1)
        df = df[df['content'].str.strip() != ""]

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[(df["timestamp"] >= start_date) & (df["timestamp"] < end_date)]
        df = df.sort_values(by="timestamp")
        df = df.drop_duplicates(subset=["timestamp", "title"], keep="first")

        df = df.reset_index(drop=True)
        df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

        # Call _summary function for news summarization
        # if not df.empty:
        #     logger.info(f"Starting LLM summarization for {len(df)} news items")
        #     df = await self._summary(df)
        #     logger.info("News summarization completed")
        # else:
        #     logger.warning("No news data to summarize")

        df['summary'] = ""
        df = df[["timestamp"] + ["title", "content", "summary"]]

        # update df to info
        info["names"] = list(sorted(df.columns))
        info["start_date"] = df["timestamp"].min()
        info["end_date"] = df["timestamp"].max()

        df.to_json(os.path.join(self.workdir, "{}.jsonl".format(self.symbol)), orient="records", lines=True)

        return info