#!/usr/bin/env bash
# vLLM service manager for local caption-attribute parsing.

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-../models/openai/gpt-oss-20b}
MODEL_NAME=${MODEL_NAME:-openai/gpt-oss-20b}
BASE_PORT=${BASE_PORT:-18000}
GPUS_STR=${GPUS:-"0 1 2 3 4 5 6"}
RUNTIME_DIR=${RUNTIME_DIR:-outputs/vllm}

read -r -a GPUS_ARRAY <<< "$GPUS_STR"

MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

pid_file() {
    local gpu=$1
    local port=$2
    echo "${RUNTIME_DIR}/vllm_gpu${gpu}_port${port}.pid"
}

log_file() {
    local gpu=$1
    local port=$2
    echo "${RUNTIME_DIR}/vllm_gpu${gpu}_port${port}.log"
}

start_vllm_service() {
    local gpu=$1
    local port=$2
    mkdir -p "$RUNTIME_DIR"

    echo "=========================================="
    echo "启动 vLLM 服务"
    echo "GPU: $gpu"
    echo "Port: $port"
    echo "Model: $MODEL_NAME"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$gpu nohup vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$port" \
        --served-model-name "$MODEL_NAME" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        > "$(log_file "$gpu" "$port")" 2>&1 &

    local pid=$!
    echo "vLLM服务已启动 (GPU $gpu, Port $port, PID: $pid)"
    echo "$pid" > "$(pid_file "$gpu" "$port")"
}

wait_for_service() {
    local port=$1
    local max_wait=300
    local waited=0

    echo "等待端口 $port 的服务就绪..."

    while [ "$waited" -lt "$max_wait" ]; do
        if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "✓ 端口 $port 的服务已就绪"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        echo "  已等待 ${waited}s..."
    done

    echo "✗ 端口 $port 的服务启动超时"
    return 1
}

check_service_status() {
    local gpu=$1
    local port=$2
    local pid_path
    pid_path=$(pid_file "$gpu" "$port")

    echo -n "GPU $gpu (Port $port): "

    if [ -f "$pid_path" ]; then
        pid=$(cat "$pid_path")
        if kill -0 "$pid" 2>/dev/null; then
            echo -n "✓ 运行中 (PID: $pid) | "
            if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
                echo "API: ✓"
            else
                echo "API: ✗"
            fi
        else
            echo "✗ 进程已停止"
        fi
    else
        echo "✗ 未启动"
    fi
}

stop_all_services() {
    echo ""
    echo "=========================================="
    echo "停止所有 vLLM 服务"
    echo "=========================================="

    for gpu_idx in "${!GPUS_ARRAY[@]}"; do
        gpu=${GPUS_ARRAY[$gpu_idx]}
        port=$((BASE_PORT + gpu_idx))
        pid_path=$(pid_file "$gpu" "$port")

        if [ -f "$pid_path" ]; then
            pid=$(cat "$pid_path")
            if kill -0 "$pid" 2>/dev/null; then
                echo "停止 GPU $gpu (Port $port, PID: $pid)"
                kill "$pid" 2>/dev/null || true
                sleep 1
            fi
            rm -f "$pid_path"
        fi
    done

    pkill -f "vllm serve.*${MODEL_PATH}" 2>/dev/null || true
    echo "✓ 所有服务已停止"
}

show_status() {
    echo ""
    echo "=========================================="
    echo "vLLM 服务状态"
    echo "=========================================="

    for gpu_idx in "${!GPUS_ARRAY[@]}"; do
        gpu=${GPUS_ARRAY[$gpu_idx]}
        port=$((BASE_PORT + gpu_idx))
        check_service_status "$gpu" "$port"
    done

    echo ""
    echo "GPU 使用情况:"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader 2>/dev/null | while IFS=',' read -r idx name temp util mem_used mem_total; do
        echo "  GPU $idx: $name | 温度: $temp | 利用率: $util | 显存: $mem_used / $mem_total"
    done
}

test_services() {
    echo ""
    echo "=========================================="
    echo "测试 vLLM 服务"
    echo "=========================================="

    all_ok=true

    for gpu_idx in "${!GPUS_ARRAY[@]}"; do
        port=$((BASE_PORT + gpu_idx))
        echo -n "测试 Port $port ... "
        response=$(curl -s "http://localhost:${port}/v1/models" || true)

        if [ -n "$response" ]; then
            echo "✓ 正常"
        else
            echo "✗ 失败"
            all_ok=false
        fi
    done

    echo ""
    if [ "$all_ok" = true ]; then
        echo "✓ 所有服务测试通过"
    else
        echo "✗ 部分服务测试失败"
        return 1
    fi
}

print_usage() {
    echo "用法: $0 [start|stop|restart|status|test]"
    echo ""
    echo "命令:"
    echo "  start   - 启动所有 vLLM 服务"
    echo "  stop    - 停止所有 vLLM 服务"
    echo "  restart - 重启所有 vLLM 服务"
    echo "  status  - 显示服务状态"
    echo "  test    - 测试服务是否正常"
    echo ""
    echo "配置:"
    echo "  模型路径: $MODEL_PATH"
    echo "  GPU设备: ${GPUS_ARRAY[*]}"
    echo "  端口范围: $BASE_PORT-$((BASE_PORT + ${#GPUS_ARRAY[@]} - 1))"
    echo "  运行目录: $RUNTIME_DIR"
}

ACTION=${1:-start}

case "$ACTION" in
    start)
        echo "=========================================="
        echo "启动 vLLM 服务"
        echo "=========================================="
        echo "配置信息:"
        echo "  模型路径: $MODEL_PATH"
        echo "  模型名称: $MODEL_NAME"
        echo "  GPU数量: ${#GPUS_ARRAY[@]}"
        echo "  GPU设备: ${GPUS_ARRAY[*]}"
        echo "  端口范围: $BASE_PORT-$((BASE_PORT + ${#GPUS_ARRAY[@]} - 1))"
        echo "  最大序列长度: $MAX_MODEL_LEN"
        echo "  最大并发数: $MAX_NUM_SEQS"
        echo "  显存利用率: $GPU_MEMORY_UTILIZATION"
        echo "  运行目录: $RUNTIME_DIR"
        echo ""

        running_count=0
        for gpu_idx in "${!GPUS_ARRAY[@]}"; do
            gpu=${GPUS_ARRAY[$gpu_idx]}
            port=$((BASE_PORT + gpu_idx))
            pid_path=$(pid_file "$gpu" "$port")
            if [ -f "$pid_path" ]; then
                pid=$(cat "$pid_path")
                if kill -0 "$pid" 2>/dev/null; then
                    running_count=$((running_count + 1))
                fi
            fi
        done

        if [ "$running_count" -gt 0 ]; then
            echo "⚠️  检测到 $running_count 个服务已在运行"
            read -p "是否停止现有服务并重新启动? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                stop_all_services
                sleep 2
            else
                echo "取消启动"
                exit 0
            fi
        fi

        echo "启动服务..."
        for gpu_idx in "${!GPUS_ARRAY[@]}"; do
            gpu=${GPUS_ARRAY[$gpu_idx]}
            port=$((BASE_PORT + gpu_idx))
            start_vllm_service "$gpu" "$port"
            sleep 2
        done

        echo ""
        echo "等待服务就绪..."

        all_ready=true
        for gpu_idx in "${!GPUS_ARRAY[@]}"; do
            port=$((BASE_PORT + gpu_idx))
            if ! wait_for_service "$port"; then
                all_ready=false
            fi
        done

        if [ "$all_ready" = true ]; then
            echo ""
            echo "✓ 所有服务启动成功"
            show_status

            echo ""
            echo "服务端点:"
            for gpu_idx in "${!GPUS_ARRAY[@]}"; do
                port=$((BASE_PORT + gpu_idx))
                echo "  http://localhost:$port/v1"
            done

            echo ""
            echo "查看日志:"
            for gpu_idx in "${!GPUS_ARRAY[@]}"; do
                gpu=${GPUS_ARRAY[$gpu_idx]}
                port=$((BASE_PORT + gpu_idx))
                echo "  tail -f $(log_file "$gpu" "$port")"
            done
        else
            echo ""
            echo "✗ 部分服务启动失败"
            echo "请查看日志文件: ${RUNTIME_DIR}/vllm_gpu*_port*.log"
            exit 1
        fi
        ;;
    stop)
        stop_all_services
        ;;
    restart)
        stop_all_services
        sleep 2
        "$0" start
        ;;
    status)
        show_status
        ;;
    test)
        test_services
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
