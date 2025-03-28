# ChatGPT Summary Plugin

一个基于 ChatGPT 的聊天记录总结插件，支持多模态图片识别功能。

## 功能特点

- 支持聊天记录总结
- 支持图片自动识别和描述
- 灵活的时间范围指定
- 自定义总结指令
- 多模态 LLM 支持
- 支持指定群聊/用户名总结（需密码验证）
- 支持模糊匹配白名单功能，更灵活地设定要记录的会话
- 支持模糊匹配指定会话并提供多结果选择功能

## 使用方法

### 基本命令

$总结 [参数]

### 支持的命令格式

- `$总结 100` - 总结最近100条消息
- `$总结 -2h 100` - 总结过去2小时内的消息，最多100条
- `$总结 -24h` - 总结过去24小时内的消息
- `$总结 100 自定义指令` - 总结最近100条消息，并按照自定义指令进行总结
- `$总结 g群名称 密码 100` - 总结指定群最近100条消息（需要密码验证，支持模糊匹配）
- `$总结 u用户名 密码 -2h` - 总结指定用户最近2小时消息（需要密码验证，支持模糊匹配）
- `$总结选择 编号 [其他参数]` - 从多个匹配结果中选择指定编号的会话进行总结

### 自定义指令说明

自定义指令可以在命令末尾添加，用于指定总结的重点和方向：

- `$总结 100 只总结和股票相关的内容` - 从最近100条消息中提取股票相关内容
- `$总结 -2h 只关注技术讨论` - 总结过去2小时内的技术讨论
- `$总结 -24h 200 请用幽默的语气总结` - 用幽默的方式总结过去24小时内最多200条消息
- `$总结 g测试群 密码 100 只总结会议内容` - 总结指定群的会议相关内容

### 模糊匹配与会话选择

当使用 `$总结 g群名称` 或 `$总结 u用户名` 命令时，插件会进行模糊匹配：

1. 如果只找到一个匹配会话，会直接使用该会话进行总结
2. 如果找到多个匹配会话，会返回编号列表让用户选择
3. 用户可以通过 `$总结选择 编号 [其他参数]` 命令选择特定的会话
4. 例如：`$总结选择 2 -24h 100` 表示选择列表中的第2个会话，总结过去24小时内最多100条消息

优先级规则：
1. 自定义指令具有最高优先级
2. 如果涉及总结，会参考默认的总结规则
3. 没有自定义指令时，执行默认的总结操作

### 配置说明

在 `config.json` 中配置以下参数：

```json
{
    "multimodal_llm_api_base": "多模态LLM API地址",
    "multimodal_llm_model": "多模态LLM模型名称",
    "multimodal_llm_api_key": "多模态LLM API密钥",
    "summary_password": "设置访问密码",
    "summary_max_tokens": 8000,
    "input_max_tokens_limit": 160000,
    "chunk_max_tokens": 16000,
    "record_all": true,
    "whitelist_groups": ["测试群", "工作群"],
    "whitelist_users": ["张三", "李四"],
    "use_fuzzy_matching": true
}
```

配置项说明：
- `multimodal_llm_*`: 多模态LLM相关配置，用于图片识别功能
- `summary_password`: 指定会话总结功能的访问密码
- `summary_max_tokens`: 总结内容的最大token数
- `input_max_tokens_limit`: 输入内容的最大token数限制
- `chunk_max_tokens`: 每个处理块的最大token数
- `record_all`: 是否记录所有会话，设为 false 时只记录白名单中的会话
- `whitelist_groups`: 群聊白名单列表
- `whitelist_users`: 私聊白名单列表
- `use_fuzzy_matching`: 是否启用模糊匹配（默认为 true），设为 false 时使用精确匹配

## 输出格式

总结内容将按以下格式输出：

```
1️⃣[Topic][热度🔥]
• 时间：月-日 时:分 - -日 时:分
• 参与者：
• 内容：
• 结论：
```

## 注意事项

1. 图片识别功能需要配置多模态 LLM 相关参数
2. 总结时间范围建议使用小时单位（如 -2h）
3. 自定义指令可以引导总结的方向和重点
4. 插件会自动过滤短命令消息（<50字符且包含$或#）
5. 指定群聊或用户名总结需要正确的访问密码
6. 访问密码未配置时无法使用指定会话功能
7. 指定群聊或用户名总结功能仅支持私聊使用，群聊中无法使用该功能
8. 白名单模糊匹配功能默认启用，可通过配置关闭
9. 模糊匹配时，只要白名单名称部分包含实际会话名称或实际会话名称包含白名单名称即可匹配成功
10. 当指定群名或用户名进行总结时，若存在多个匹配结果，可通过选择命令指定要总结的会话

## 更新日志

### v1.6.3
- 新增指定会话总结时的模糊匹配功能，支持多结果选择

### v1.6.2
- 新增白名单模糊匹配功能，更灵活地设定要记录的会话
- 新增配置选项 use_fuzzy_matching，可切换精确匹配和模糊匹配模式

### v1.4
- 移除直接调用 OpenAI API 进行总结的功能
- 改为通过 bot 处理总结请求
- 优化总结处理流程，添加处理中提示

### v1.3
- 新增指定群聊/用户名总结功能
- 添加密码验证机制
- 限制指定会话总结功能仅支持私聊使用

## 许可证

MIT License

本插件基于深蓝老大的代码修改而来，感谢深蓝大佬的贡献。

## 打赏支持

如果您觉得这个插件对您有帮助，欢迎扫描下方二维码进行打赏支持，让我能够持续改进和开发更多实用功能。

![微信打赏码](https://github.com/sofs2005/difytask/raw/main/img/wx.png?raw=true)