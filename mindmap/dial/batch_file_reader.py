import asyncio
import os

import aiohttp

from mindmap.dial.client import DialClient
from mindmap.utils.file_cache import FileCache
from mindmap.utils.logger_config import logger

BATCH_READ_REQUESTS_LIMIT = int(os.getenv("BATCH_READ_REQUESTS_LIMIT", 300))


class BatchFileReader:
    files: list[str]
    client: DialClient
    cache: FileCache | None
    cache_hits = 0
    _cache_hits_lock = asyncio.Lock()

    def __init__(self, client: DialClient, cache: FileCache | None = None):
        self.files = []
        self.client = client
        self.cache = cache

    def add_file(self, file: str):
        self.files.append(file)

    async def read_file(
        self, sem: asyncio.Semaphore, file: str, session: aiohttp.ClientSession
    ):
        async with sem:
            file_name = self.client.make_url(file)

            cache_value = (
                self.cache.get(file_name) if self.cache is not None else None
            )

            if cache_value:
                async with self._cache_hits_lock:
                    self.cache_hits += 1

                return (file, cache_value)
            else:
                value = (
                    await self.client.read_file_by_name_and_etag(
                        file, session=session, batch=True
                    )
                )[0]

                if self.cache is not None:
                    self.cache.set(file_name, value)

                return (file, value)

    async def read(self):
        if len(self.files) == 0:
            return []

        logger.info(f"Starting batch file read (0/{len(self.files)})")

        sem = asyncio.Semaphore(BATCH_READ_REQUESTS_LIMIT)

        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.ensure_future(self.read_file(sem, file, session))
                for file in self.files
            ]

            result = await asyncio.gather(*tasks)

        logger.info(
            f"Finished batch file read ({len(self.files)}/{len(self.files)}, cache hits: {self.cache_hits})"
        )

        return result


class BatchRawFileReader:
    files: list[str]
    client: DialClient

    def __init__(self, client: DialClient):
        self.files = []
        self.client = client

    def add_file(self, file: str):
        self.files.append(file)

    async def read_file(self, file: str):
        return (file, await self.client.read_raw_file_by_url(file))

    async def read(self):
        tasks = [
            asyncio.ensure_future(self.read_file(file)) for file in self.files
        ]
        return await asyncio.gather(*tasks)
