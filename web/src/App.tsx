export default function App() {
  return (
    <main className="app-shell">
      <header className="hero">
        <p className="eyebrow">InferenceSimulator · M3</p>
        <h1>推理服务仿真器</h1>
        <p className="hero-description">
          配置单 GPU 推理负载，并查看仿真指标与请求生命周期。
        </p>
      </header>

      <section className="workspace-card" aria-labelledby="workspace-title">
        <h2 id="workspace-title">前端工作区已就绪</h2>
        <p>配置表单与仿真结果将在后续模块中逐步提供。</p>
      </section>
    </main>
  );
}
