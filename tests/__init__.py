"""unittest 测试包。

每个测试模块都内联了将 ``src/`` 加入 ``sys.path`` 的引导，因此无论是否执行过
``pip install -e .``、无论使用 ``unittest discover``（仓库根或 ``-s tests``）
还是 ``python -m unittest tests.<module>`` 都可以运行。
"""
