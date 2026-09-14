import asyncio
from collections.abc import Awaitable, Iterable
from typing import TypeVar

T = TypeVar("T")


async def gather_limited(coros: Iterable[Awaitable[T]], limit: int = 10) -> list[T]:
    semaphore = asyncio.Semaphore(limit)
    results: list[T] = []

    async def _wrap(coro: Awaitable[T]) -> None:
        async with semaphore:
            res = await coro
            results.append(res)

    await asyncio.gather(*[_wrap(c) for c in coros])
    return results
