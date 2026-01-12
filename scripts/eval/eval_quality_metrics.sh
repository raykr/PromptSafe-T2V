#!/bin/bash

# T2V安全Adapter防御效果评估 - 批量运行所有模型和所有类别
# 用法: bash scripts/eval/eval_quality_metrics.sh

# 注意：不使用 set -e，因为我们需要并行执行时即使某个任务失败也继续

# 配置路径
BASE_DIR="/home/raykr/projects/PromptSafe-T2V"
PROMPT_TINY_DIR="${BASE_DIR}/datasets/test/tiny"
PROMPT_BENIGN="${BASE_DIR}/datasets/test/benign.csv"
OUTPUT_BASE_DIR="${BASE_DIR}/evaluation_results"

# 模型列表（目录名称）
MODELS=("cogvideox-2b" "CogVideoX-5b" "CogVideoX1.5-5B")

# 类别列表（tiny目录下的类别）
TINY_CATEGORIES=("disturbing" "political" "sexual" "violent")
# benign类别需要特殊处理（不在tiny目录下）

# 设备配置
DEVICE="cuda"

# 并行配置
PARALLEL_JOBS=5  # 每个模型同时运行的类别数（5个类别可以全部并行）

# 视频文件命名模式
BASELINE_PATTERN="adapter_{:03d}_raw.mp4"
DEFENDED_PATTERN="multi_{:03d}_safe.mp4"

echo "======================================================================"
echo "T2V安全Adapter防御效果评估 - 批量运行所有模型和所有类别"
echo "======================================================================"
echo "基础目录: ${BASE_DIR}"
echo "Prompt目录 (tiny): ${PROMPT_TINY_DIR}"
echo "Prompt文件 (benign): ${PROMPT_BENIGN}"
echo "输出目录: ${OUTPUT_BASE_DIR}"
echo "设备: ${DEVICE}"
echo "并行度: ${PARALLEL_JOBS} (每个模型的类别并行执行)"
echo "模型: ${MODELS[@]}"
echo "类别: ${TINY_CATEGORIES[@]} + benign"
echo "======================================================================"
echo ""

# 创建输出基础目录
mkdir -p "${OUTPUT_BASE_DIR}"

# 记录开始时间
START_TIME=$(date +%s)

# 统计信息
TOTAL_TINY_CATEGORIES=${#TINY_CATEGORIES[@]}
TOTAL_TASKS=$((${#MODELS[@]} * (${#TINY_CATEGORIES[@]} + 1)))  # +1 for benign
COMPLETED_TASKS=0
FAILED_TASKS=0

# 遍历每个模型
for model in "${MODELS[@]}"; do
    VIDEO_BASE_DIR="${BASE_DIR}/out/${model}/tiny"
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║  模型: ${model}"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # 检查模型目录是否存在
    if [ ! -d "${VIDEO_BASE_DIR}" ]; then
        echo "⚠️  警告: 模型tiny目录不存在 - ${VIDEO_BASE_DIR}"
        echo "   继续检查benign目录..."
    fi
    
    # 创建临时日志目录
    LOG_DIR="${OUTPUT_BASE_DIR}/.logs/${model}"
    mkdir -p "${LOG_DIR}"
    
    # 存储后台进程PID
    PIDS=()
    
    echo "🚀 启动并行评估任务..."
    echo ""
    
    # 1. 遍历tiny目录下的类别（并行执行）
    for category in "${TINY_CATEGORIES[@]}"; do
        # 构建路径（tiny类别）
        PROMPT_CSV="${PROMPT_TINY_DIR}/${category}.csv"
        BASELINE_DIR="${VIDEO_BASE_DIR}/${category}/baseline"
        DEFENDED_DIR="${VIDEO_BASE_DIR}/${category}/multi_defense"
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/${model}/${category}"
        LOG_FILE="${LOG_DIR}/${category}.log"
        
        # 检查文件是否存在
        if [ ! -f "${PROMPT_CSV}" ]; then
            echo "⚠️  警告: Prompt文件不存在 - ${PROMPT_CSV}"
            echo "   跳过: [${model}] ${category}"
            FAILED_TASKS=$((FAILED_TASKS + 1))
            continue
        fi
        
        if [ ! -d "${BASELINE_DIR}" ]; then
            echo "⚠️  警告: Baseline目录不存在 - ${BASELINE_DIR}"
            echo "   跳过: [${model}] ${category}"
            FAILED_TASKS=$((FAILED_TASKS + 1))
            continue
        fi
        
        if [ ! -d "${DEFENDED_DIR}" ]; then
            echo "⚠️  警告: Defended目录不存在 - ${DEFENDED_DIR}"
            echo "   跳过: [${model}] ${category}"
            FAILED_TASKS=$((FAILED_TASKS + 1))
            continue
        fi
        
        # 创建输出目录
        mkdir -p "${OUTPUT_DIR}"
        
        # 在后台运行评估任务（使用stdbuf确保实时写入日志，不输出到终端）
        (
            {
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始评估 [${model}] ${category}"
                echo "📝 Prompt文件: ${PROMPT_CSV}"
                echo "📹 Baseline目录: ${BASELINE_DIR}"
                echo "🛡️  Defended目录: ${DEFENDED_DIR}"
                echo "💾 输出目录: ${OUTPUT_DIR}"
                echo ""
                
                # 使用 -u 参数禁用Python输出缓冲，stdbuf设置行缓冲，确保实时写入日志
                stdbuf -oL -eL python3 -u "${BASE_DIR}/t2v_evaluation_metrics.py" \
                    --prompt_csv "${PROMPT_CSV}" \
                    --baseline_dir "${BASELINE_DIR}" \
                    --defended_dir "${DEFENDED_DIR}" \
                    --output_dir "${OUTPUT_DIR}" \
                    --device "${DEVICE}" \
                    --baseline_pattern "${BASELINE_PATTERN}" \
                    --defended_pattern "${DEFENDED_PATTERN}"
                
                EXIT_CODE=$?
                if [ ${EXIT_CODE} -eq 0 ]; then
                    echo ""
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ [${model}] ${category} 评估完成"
                    echo "SUCCESS" > "${LOG_DIR}/${category}.status"
                else
                    echo ""
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ [${model}] ${category} 评估失败 (退出码: ${EXIT_CODE})"
                    echo "FAILED" > "${LOG_DIR}/${category}.status"
                fi
            } 2>&1 | stdbuf -oL -eL cat > "${LOG_FILE}"
        ) &
        
        PID=$!
        PIDS+=(${PID})
        echo "  → 启动任务 [${model}] ${category} (PID: ${PID})"
    done
    
    # 2. 处理benign类别（也并行执行）
    BENIGN_VIDEO_DIR="${BASE_DIR}/out/${model}/benign"
    BASELINE_DIR="${BENIGN_VIDEO_DIR}/baseline"
    DEFENDED_DIR="${BENIGN_VIDEO_DIR}/multi_defense"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${model}/benign"
    LOG_FILE="${LOG_DIR}/benign.log"
    
    if [ -f "${PROMPT_BENIGN}" ] && [ -d "${BASELINE_DIR}" ] && [ -d "${DEFENDED_DIR}" ]; then
        mkdir -p "${OUTPUT_DIR}"
        
        (
            {
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始评估 [${model}] benign"
                echo "📝 Prompt文件: ${PROMPT_BENIGN}"
                echo "📹 Baseline目录: ${BASELINE_DIR}"
                echo "🛡️  Defended目录: ${DEFENDED_DIR}"
                echo "💾 输出目录: ${OUTPUT_DIR}"
                echo ""
                
                # 使用 -u 参数禁用Python输出缓冲，stdbuf设置行缓冲，确保实时写入日志
                stdbuf -oL -eL python3 -u "${BASE_DIR}/t2v_evaluation_metrics.py" \
                    --prompt_csv "${PROMPT_BENIGN}" \
                    --baseline_dir "${BASELINE_DIR}" \
                    --defended_dir "${DEFENDED_DIR}" \
                    --output_dir "${OUTPUT_DIR}" \
                    --device "${DEVICE}" \
                    --baseline_pattern "${BASELINE_PATTERN}" \
                    --defended_pattern "${DEFENDED_PATTERN}"
                
                EXIT_CODE=$?
                if [ ${EXIT_CODE} -eq 0 ]; then
                    echo ""
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ [${model}] benign 评估完成"
                    echo "SUCCESS" > "${LOG_DIR}/benign.status"
                else
                    echo ""
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ [${model}] benign 评估失败 (退出码: ${EXIT_CODE})"
                    echo "FAILED" > "${LOG_DIR}/benign.status"
                fi
            } 2>&1 | stdbuf -oL -eL cat > "${LOG_FILE}"
        ) &
        
        PID=$!
        PIDS+=(${PID})
        echo "  → 启动任务 [${model}] benign (PID: ${PID})"
    else
        echo "  ⚠️  跳过 [${model}] benign (文件或目录不存在)"
        FAILED_TASKS=$((FAILED_TASKS + 1))
    fi
    
    # 等待所有后台任务完成
    echo ""
    echo "⏳ 等待所有任务完成 (共 ${#PIDS[@]} 个任务)..."
    echo ""
    
    for PID in "${PIDS[@]}"; do
        wait ${PID}
    done
    
    # 统计结果
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 [${model}] 任务完成情况:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    for category in "${TINY_CATEGORIES[@]}"; do
        STATUS_FILE="${LOG_DIR}/${category}.status"
        if [ -f "${STATUS_FILE}" ]; then
            STATUS=$(cat "${STATUS_FILE}")
            if [ "${STATUS}" = "SUCCESS" ]; then
                echo "  ✅ [${model}] ${category}: 成功"
                COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
            else
                echo "  ❌ [${model}] ${category}: 失败"
                FAILED_TASKS=$((FAILED_TASKS + 1))
            fi
        fi
    done
    
    STATUS_FILE="${LOG_DIR}/benign.status"
    if [ -f "${STATUS_FILE}" ]; then
        STATUS=$(cat "${STATUS_FILE}")
        if [ "${STATUS}" = "SUCCESS" ]; then
            echo "  ✅ [${model}] benign: 成功"
            COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
        else
            echo "  ❌ [${model}] benign: 失败"
            FAILED_TASKS=$((FAILED_TASKS + 1))
        fi
    fi
    
    echo ""
    echo "📋 详细日志保存在: ${LOG_DIR}"
    echo ""
    
done

# 计算总耗时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "======================================================================"
echo "🎉 所有模型和类别评估完成！"
echo "======================================================================"
echo "总任务数: ${TOTAL_TASKS}"
echo "成功完成: ${COMPLETED_TASKS}"
echo "失败/跳过: ${FAILED_TASKS}"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo "结果保存在: ${OUTPUT_BASE_DIR}"
echo ""
echo "结果目录结构:"
for model in "${MODELS[@]}"; do
    MODEL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${model}"
    if [ -d "${MODEL_OUTPUT_DIR}" ]; then
        echo "  📁 ${model}/"
        # tiny类别
        for category in "${TINY_CATEGORIES[@]}"; do
            CATEGORY_OUTPUT_DIR="${MODEL_OUTPUT_DIR}/${category}"
            if [ -d "${CATEGORY_OUTPUT_DIR}" ]; then
                echo "    └── ${category}/"
            fi
        done
        # benign类别
        BENIGN_OUTPUT_DIR="${MODEL_OUTPUT_DIR}/benign"
        if [ -d "${BENIGN_OUTPUT_DIR}" ]; then
            echo "    └── benign/"
        fi
    fi
done
echo ""
echo "📋 所有任务的详细日志保存在: ${OUTPUT_BASE_DIR}/.logs/"
echo "======================================================================"
