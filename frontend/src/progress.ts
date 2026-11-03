import type { RunEvent } from './types';

export interface ProgressStep {
  id: string;
  title: string;
  detail?: string;
  state: 'running' | 'complete' | 'error' | 'neutral';
  status: string;
}

const toolNames: Record<string, string> = {
  profile_csv: '读取 CSV 数据',
  set_task_plan: '制定交付计划',
  search_materials: '检索任务资料',
  delegate_analysis: '委派代码 Agent 分析',
  read_artifact: '核对分析成果',
  read_skill: '读取分析技能',
  ask_user: '确认关键口径',
};

function resultState(status: unknown): Pick<ProgressStep, 'state' | 'status'> {
  if (status === 'completed' || status === 'success') return { state: 'complete', status: '完成' };
  if (status === 'error' || status === 'failed') return { state: 'error', status: '失败' };
  if (status === 'awaiting_input') return { state: 'neutral', status: '等待回复' };
  if (status === 'cancelled') return { state: 'neutral', status: '已停止' };
  if (status === 'interrupted') return { state: 'neutral', status: '已中断' };
  return { state: 'neutral', status: '已返回' };
}

/** Summarize genuine host events; never infer a successful step from model prose. */
export function summarizeProgress(events: RunEvent[], active: boolean, awaitingInput = false): ProgressStep[] {
  const steps: ProgressStep[] = [];
  const byId = new Map<string, ProgressStep>();
  const pendingRetries = new Map<string, ProgressStep>();
  const update = (step: ProgressStep) => {
    const previous = byId.get(step.id);
    if (previous) Object.assign(previous, step);
    else { byId.set(step.id, step); steps.push(step); }
  };
  for (const event of events) {
    const payload = event.payload;
    const started = { state: active && !awaitingInput ? 'running' as const : 'neutral' as const, status: awaitingInput ? '等待回复' : active ? '执行中' : '仅有开始记录' };
    if (event.type === 'running') {
      update({ id: `start:${event.seq}`, title: typeof payload.resumed_question_id === 'string' ? '已接收回复，继续分析' : '宿主已接收任务，启动分析', state: 'neutral', status: '已开始' });
    } else if (event.type === 'skill_loaded') {
      const readId = payload.parent_call_id ? `${payload.parent_call_id}:${payload.tool_call_id ?? event.seq}` : String(event.seq);
      update({ id: `skill:${payload.role}:${readId}`, title: '已读取分析技能', detail: String(payload.skill_id ?? ''), state: 'complete', status: '已加载' });
    } else if (event.type === 'cancelling') {
      update({ id: `cancel:${event.seq}`, title: '已请求停止当前执行并检查清理结果', state: active ? 'running' : 'neutral', status: active ? '停止中' : '已请求停止' });
    } else if (event.type === 'model_start' || event.type === 'model_end' || event.type === 'model_retry') {
      const role = payload.role === 'analysis' ? '代码 Agent' : payload.role === 'main' ? '主 Agent' : 'Agent';
      const parent = typeof payload.parent_call_id === 'string' ? payload.parent_call_id : 'main';
      const scope = `${String(payload.role)}:${parent}`;
      const call = typeof payload.call === 'number' && Number.isFinite(payload.call) && payload.call > 0 ? payload.call : null;
      if (event.type === 'model_retry') {
        const reason = payload.error_code === 'model_output_truncated' ? '模型响应截断' : payload.error_code === 'invalid_tool_arguments' ? '工具参数无效' : '模型响应未完成';
        const step: ProgressStep = { id: `model-retry:${scope}:${call ?? event.seq}`, title: `${role} · ${reason}，进行一次修正`, state: active && !awaitingInput ? 'running' : 'neutral', status: active && !awaitingInput ? '修正中' : '已请求修正' };
        update(step); pendingRetries.set(scope, step);
      } else {
        const truncated = payload.finish_reason === 'length';
        const elapsed = typeof payload.elapsed_seconds === 'number' && Number.isFinite(payload.elapsed_seconds) ? `耗时 ${payload.elapsed_seconds.toFixed(1)} 秒` : '';
        const outcome = payload.status === 'error' ? { state: 'error' as const, status: '请求失败' } : truncated ? { state: 'neutral' as const, status: '响应截断' } : { state: 'neutral' as const, status: '已返回' };
        update({ id: `model:${scope}:${call ?? event.seq}`, title: `${role} · ${call ? `第 ${call} 次` : ''}模型生成`, ...(event.type === 'model_start' ? { ...started, status: active && !awaitingInput ? '生成中' : started.status } : outcome), detail: event.type === 'model_end' ? elapsed || undefined : undefined });
        if (event.type === 'model_end') {
          const retry = pendingRetries.get(scope);
          if (retry) { update({ ...retry, state: outcome.state, status: truncated ? '仍被截断' : outcome.status }); pendingRetries.delete(scope); }
        }
      }
    } else if (event.type === 'verification_start' || event.type === 'verification_end') {
      const verification=payload.verification as {passed?: boolean} | undefined;
      update({id:`verify:${String(payload.execution_id)}`,title:'独立核对结果',...(event.type==='verification_start' ? started : resultState(verification?.passed ? 'completed':'error')),detail:event.type==='verification_end' ? '核验覆盖与未检查项见报告明细' : undefined});
    } else if (event.type === 'tool_start' || event.type === 'tool_end') {
      const name = typeof payload.name === 'string' ? payload.name : '工具';
      update({ id: `tool:${typeof payload.id === 'string' ? payload.id : event.seq}`, title: toolNames[name] ?? `调用 ${name}`, ...(event.type === 'tool_end' ? resultState(payload.status) : started) });
    } else if (event.type === 'execution_start' || event.type === 'execution_end') {
      const attempt = typeof payload.attempt === 'number' ? payload.attempt : 1;
      const record = payload.record && typeof payload.record === 'object' ? payload.record as Record<string, unknown> : {};
      const outcome = record.status ?? payload.status;
      const missing = Array.isArray(payload.missing_deliverables) ? payload.missing_deliverables.filter((value): value is string => typeof value === 'string').join('、') : '';
      const incomplete = payload.delivery_status === 'error' && outcome === 'completed';
      const duration = typeof record.elapsed_seconds === 'number' ? `耗时 ${record.elapsed_seconds.toFixed(1)} 秒` : '';
      const exit = typeof record.exit_code === 'number' ? `退出码 ${record.exit_code}` : '';
      const failed = outcome === 'error';
      const error = record.error && typeof record.error === 'object' ? record.error as Record<string, unknown> : {};
      const stderrObject = record.stderr && typeof record.stderr === 'object' ? record.stderr as Record<string, unknown> : {};
      const stderr = typeof record.stderr === 'string' ? record.stderr : typeof stderrObject.text === 'string' ? stderrObject.text : '';
      const stderrLine = stderr.split(/\r?\n/).filter(line => line.trim()).at(-1);
      const failure = failed ? String(stderrLine || error.message || error.code || '').replace(/\u001b\[[0-9;]*m/g, '').slice(0, 160) : '';
      update({ id: `python:${typeof payload.tool_call_id === 'string' ? payload.tool_call_id : `${payload.parent_call_id}:${attempt}`}`, title: `代码 Agent · 第 ${attempt} 次 Python ${attempt > 1 ? '修正执行' : '执行'}`, ...(event.type === 'execution_end' ? incomplete ? { state: 'error' as const, status: '成果缺失' } : resultState(outcome) : started), detail: [exit, duration, incomplete ? `代码已运行，缺少成果${missing ? `：${missing}` : ''}` : failure, failed && stderrObject.truncated ? '日志预览已截断' : ''].filter(Boolean).join(' · ') });
    } else if (event.type === 'retrieval_result' || event.type === 'material_search') {
      const count = Array.isArray(payload.matches) ? payload.matches.length : null;
      update({ id: `search:${event.seq}`, title: '资料检索结果', detail: count === null ? undefined : `命中 ${count} 个真实片段`, ...resultState(payload.status) });
    } else if (event.type === 'validated') {
      update({ id: `validation:${event.seq}`, title: '核对执行证据', ...resultState(payload.status), detail: payload.status === 'completed' ? '等待成果保存与登记' : undefined });
    } else if (event.type === 'saving' || event.type === 'artifacts_saved') {
      update({ id: `save:${event.seq}`, title: event.type === 'saving' ? '保存并登记成果' : '已保存本次执行的部分成果', state: event.type === 'saving' && active ? 'running' : 'neutral', status: event.type === 'saving' && active ? '保存中' : '已记录' });
    } else if (['completed', 'error', 'cancelled', 'interrupted', 'awaiting_input'].includes(event.type)) {
      const titles: Record<string, string> = { completed: '成果已核验、保存并登记', error: '任务未完成，保留已有成果', cancelled: '任务已停止', interrupted: '执行已中断', awaiting_input: '需要你的回复才能继续' };
      update({ id: `state:${event.seq}`, title: titles[event.type], ...resultState(event.type) });
    }
  }
  return steps;
}
