import asyncio
import time
import aiohttp

# 1. 基本协程
async def basic_coroutine(name, delay):
    """基本协程函数"""
    print(f"协程 {name} 开始")
    await asyncio.sleep(delay)  # 模拟I/O操作
    print(f"协程 {name} 结束")
    return f"{name} completed"

async def run_basic_coroutines():
    """运行多个协程"""
    # 方式1: 使用await顺序执行
    result1 = await basic_coroutine("Task1", 2)
    print(result1)
    
    # 方式2: 并发执行
    task1 = asyncio.create_task(basic_coroutine("Task2", 1))
    task2 = asyncio.create_task(basic_coroutine("Task3", 3))
    
    await task1
    await task2
    
    # 方式3: 使用gather
    results = await asyncio.gather(
        basic_coroutine("Task4", 1),
        basic_coroutine("Task5", 2),
        basic_coroutine("Task6", 1.5)
    )
    print(f"所有结果: {results}")

asyncio.run(run_basic_coroutines())