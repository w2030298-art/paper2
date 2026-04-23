#!/bin/bash
#
# 飞书聊天控制 Claude Code 桥接脚本
# 用法: ./lark-claude-bridge.sh
#
# 需要先开通权限:
# 1. 飞书开发者后台 → 事件与回调 → 订阅方式选"长连接" → 添加 im.message.receive_v1
# 2. 开通 im:message:receive_as_bot 权限
#

set -e

# 配置
COMMAND_PREFIX="/"  # 命令前缀，只有以此前缀开头的消息才会触发
MAX_RESPONSE_LENGTH=8000  # 最大回复长度

echo "🚀 启动飞书-Claude 桥接服务..."
echo "   监听命令: ${COMMAND_PREFIX}<command>"
echo "   按 Ctrl+C 停止"
echo ""

# 获取 bot 信息
BOT_INFO=$(lark-cli im +chat-messages-list --chat-id "oc_test" --page-limit 1 --as bot 2>/dev/null || echo "{}")
BOT_OPEN_ID=""

# 监听消息并处理
lark-cli event +subscribe \
  --event-types im.message.receive_v1 \
  --compact \
  --quiet \
  --as bot \
  2>&1 | while IFS= read -r line; do
    # 解析消息
    content=$(echo "$line" | jq -r '.content // empty' 2>/dev/null || echo "")
    message_id=$(echo "$line" | jq -r '.message_id // empty' 2>/dev/null || echo "")
    chat_id=$(echo "$line" | jq -r '.chat_id // empty' 2>/dev/null || echo "")
    sender_id=$(echo "$line" | jq -r '.sender_id // empty' 2>/dev/null || echo "")
    chat_type=$(echo "$line" | jq -r '.chat_type // empty' 2>/dev/null || echo "")

    # 跳过空消息或 bot 自己的消息
    [[ -z "$content" ]] && continue
    [[ "$sender_id" == "$BOT_OPEN_ID" ]] && continue

    # 检查命令前缀
    if [[ "$content" == "${COMMAND_PREFIX}"* ]]; then
      command="${content#"$COMMAND_PREFIX"}"
      command=$(echo "$command" | xargs)  # 去除首尾空格

      echo "[$(date '+%H:%M:%S')] 收到命令: $command"

      # 执行 Claude Code
      response=$(claude -p "$command" < /dev/null 2>&1) || response="执行出错"

      # 截断过长回复
      if [ ${#response} -gt $MAX_RESPONSE_LENGTH ]; then
        response="${response:0:$MAX_RESPONSE_LENGTH}... [内容过长已截断]"
      fi

      # 发送回复
      reply_content=$(jq -n --arg t "$response" '{msg_type:"text",content:({text:$t}|tojson)}')
      lark-cli api POST "/open-apis/im/v1/messages/$message_id/reply" \
        --data "$reply_content" \
        --as bot \
        --format data \
        > /dev/null 2>&1

      echo "[$(date '+%H:%M:%S')] 已回复"
    fi
  done
