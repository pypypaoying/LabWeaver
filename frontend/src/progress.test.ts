import { describe, expect, it } from 'vitest';
import { summarizeProgress } from './progress';
import type { RunEvent } from './types';

function event(seq: number, type: string, payload: Record<string, unknown>): RunEvent {
  return { version: 1, conversation_id: 'c', run_id: 'r', seq, type, payload };
}

describe('genuine central execution progress', () => {
  it('pairs tool starts and results by real call ID without duplicate steps', () => {
    const steps = summarizeProgress([
      event(1, 'tool_start', { name: 'profile_csv', id: 'p' }),
      event(2, 'tool_start', { name: 'profile_csv', id: 'p' }),
      event(3, 'tool_end', { name: 'profile_csv', id: 'p', status: 'completed' }),
      event(4, 'tool_start', { name: 'delegate_analysis', id: 'a' }),
    ], true);
    expect(steps).toHaveLength(2);
    expect(steps[0]).toMatchObject({ title: '读取 CSV 数据', state: 'complete', status: '完成' });
    expect(steps[1]).toMatchObject({ state: 'running', status: '执行中' });
  });

  it('uses actual code execution status, exit code and elapsed time', () => {
    const steps = summarizeProgress([
      event(1, 'execution_start', { tool_call_id: 'python', attempt: 2, code: 'private long code' }),
      event(2, 'execution_end', { tool_call_id: 'python', attempt: 2, status: 'completed', record: { status: 'error', exit_code: 1, elapsed_seconds: 2.57 } }),
    ], true);
    expect(steps).toHaveLength(1);
    expect(steps[0]).toMatchObject({ title: '代码 Agent · 第 2 次 Python 修正执行', state: 'error', status: '失败', detail: '退出码 1 · 耗时 2.6 秒' });
    expect(JSON.stringify(steps)).not.toContain('private long code');
  });

  it('never treats model prose or an unknown tool return status as success', () => {
    expect(summarizeProgress([event(1, 'answer_delta', { text: '代码执行成功' })], false)).toEqual([]);
    expect(summarizeProgress([event(1, 'tool_end', { name: 'read_artifact', id: 'r', status: 'unknown' })], false)[0]).toMatchObject({ state: 'neutral', status: '已返回' });
  });

  it('shows only a bounded real stderr summary and keeps full code out of chat progress', () => {
    const steps = summarizeProgress([event(1, 'execution_end', { tool_call_id: 'p', attempt: 1, status: 'error', code: 'entire generated code', record: { status: 'error', stderr: `Traceback\nFile generated.py\nValueError: ${'x'.repeat(220)}` } })], false);
    expect(steps[0].detail).toHaveLength(160);
    expect(steps[0].detail).toMatch(/^ValueError:/);
    expect(steps[0].detail).not.toContain('File generated.py');
    expect(JSON.stringify(steps)).not.toContain('entire generated code');
  });

  it('handles the actual core stderr object and discloses a truncated log preview', () => {
    const steps = summarizeProgress([event(1, 'execution_end', { tool_call_id: 'p', attempt: 1, status: 'error', record: { status: 'error', stderr: { text: 'Traceback\n  assert actual == expected\nAssertionError: unexpected total', truncated: true }, error: { message: 'Python exited unsuccessfully' } } })], false);
    expect(steps[0].detail).toBe('AssertionError: unexpected total · 日志预览已截断');
    expect(steps[0].detail).not.toContain('assert actual == expected');
    expect(steps[0].detail).not.toContain('Python exited unsuccessfully');
    expect(summarizeProgress([event(1, 'interrupted', {})], false)[0].status).toBe('已中断');
    expect(summarizeProgress([event(1, 'awaiting_input', {})], true)[0].status).toBe('等待回复');
  });

  it('keeps pending tools paused and final task completion separate from core validation', () => {
    expect(summarizeProgress([event(1, 'tool_start', { name: 'profile_csv', id: 'p' })], true, true)[0]).toMatchObject({ state: 'neutral', status: '等待回复' });
    const steps = summarizeProgress([event(1, 'validated', { status: 'completed' }), event(2, 'saving', {}), event(3, 'completed', {})], false);
    expect(steps[0].detail).toBe('等待成果保存与登记');
    expect(steps[2].title).toBe('成果已核验、保存并登记');
  });

  it('pairs actual model requests by role, delegation and call, without reading generation text', () => {
    const steps = summarizeProgress([
      event(1, 'model_start', { role: 'main', call: 1 }),
      event(2, 'model_end', { role: 'main', call: 1, status: 'completed', elapsed_seconds: 2.56, finish_reason: 'tool_calls', reasoning: 'private reasoning' }),
      event(3, 'model_start', { role: 'analysis', parent_call_id: 'first-delegation', call: 1 }),
      event(4, 'model_end', { role: 'analysis', parent_call_id: 'first-delegation', call: 1, status: 'completed', elapsed_seconds: 3 }),
      event(5, 'model_start', { role: 'analysis', parent_call_id: 'second-delegation', call: 1 }),
    ], true);
    expect(steps).toHaveLength(3);
    expect(steps[0]).toMatchObject({ title: '主 Agent · 第 1 次模型生成', status: '已返回', detail: '耗时 2.6 秒' });
    expect(steps[2]).toMatchObject({ title: '代码 Agent · 第 1 次模型生成', state: 'running', status: '生成中' });
    expect(JSON.stringify(steps)).not.toContain('private reasoning');
  });

  it('shows a real truncated response and one correction, without keeping a finished retry spinning', () => {
    const steps = summarizeProgress([
      event(1, 'model_end', { role: 'analysis', parent_call_id: 'd', call: 1, status: 'completed', finish_reason: 'length', elapsed_seconds: 12 }),
      event(2, 'model_retry', { role: 'analysis', parent_call_id: 'd', call: 1, error_code: 'model_output_truncated' }),
      event(3, 'model_start', { role: 'analysis', parent_call_id: 'd', call: 2 }),
      event(4, 'model_end', { role: 'analysis', parent_call_id: 'd', call: 2, status: 'completed', finish_reason: 'tool_calls', elapsed_seconds: 7 }),
    ], true);
    expect(steps[0]).toMatchObject({ state: 'neutral', status: '响应截断' });
    expect(steps[1]).toMatchObject({ title: '代码 Agent · 模型响应截断，进行一次修正', state: 'neutral', status: '已返回' });
    expect(steps[2].detail).toBe('耗时 7.0 秒');
  });

  it('uses the actual retry reason and does not turn a model return into task completion', () => {
    const steps = summarizeProgress([event(1, 'model_retry', { role: 'main', call: 2, error_code: 'invalid_tool_arguments' })], true);
    expect(steps[0]).toMatchObject({ title: '主 Agent · 工具参数无效，进行一次修正', state: 'running', status: '修正中' });
    expect(summarizeProgress([event(1, 'model_end', { role: 'main', call: 3, status: 'error' })], false)[0]).toMatchObject({ state: 'error', status: '请求失败' });
  });

  it('records actual skill reads by delegation without treating them as completed analysis', () => {
    const steps = summarizeProgress([
      event(1, 'skill_loaded', {role:'analysis',parent_call_id:'first',tool_call_id:'read-1',skill_id:'statistical-analysis'}),
      event(2, 'skill_loaded', {role:'analysis',parent_call_id:'first',tool_call_id:'read-1',skill_id:'statistical-analysis'}),
      event(3, 'skill_loaded', {role:'analysis',parent_call_id:'second',tool_call_id:'read-1',skill_id:'data-visualization'}),
    ], true);
    expect(steps).toHaveLength(2);
    expect(steps[0]).toMatchObject({title:'已读取分析技能',status:'已加载',detail:'statistical-analysis'});
    expect(steps[1].detail).toBe('data-visualization');
    const withoutParent = summarizeProgress([
      event(1, 'skill_loaded', {role:'analysis',tool_call_id:'read-1',skill_id:'statistical-analysis'}),
      event(2, 'skill_loaded', {role:'analysis',tool_call_id:'read-1',skill_id:'data-visualization'}),
    ], true);
    expect(withoutParent).toHaveLength(2);
  });

  it('distinguishes a real successful process from missing task deliverables', () => {
    const steps = summarizeProgress([event(1, 'execution_end', { tool_call_id: 'p', attempt: 1, delivery_status: 'error', missing_deliverables: ['comparison-table', 'trend-chart'], record: { status: 'completed', exit_code: 0, elapsed_seconds: 2 } })], true);
    expect(steps[0]).toMatchObject({ state: 'error', status: '成果缺失', detail: '退出码 0 · 耗时 2.0 秒 · 代码已运行，缺少成果：comparison-table、trend-chart' });
    expect(steps[0].status).not.toBe('完成');
  });
});
