# CPU示例
`python examples/decimal_demo.py` 调用项目实际的Decimal工具，核对引用并计算0.15。
`sample_messages.jsonl` 展示一题一条messages的外层格式；不是完整训练发布集。
正式训练使用成对的prompt/completion与metadata，并通过manifest与verified校验。
阶段输出契约见[数据格式](../docs/DATA.md)。示例不能直接冒充经过全流程审查的训练数据。
