const studentCountEl = document.getElementById("studentCount");
const avgAttentionEl = document.getElementById("avgAttention");
const lowAttentionEl = document.getElementById("lowAttention");
const rowsEl = document.getElementById("rows");
const lastUpdatedEl = document.getElementById("lastUpdated");

function esc(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function render(snapshot) {
  studentCountEl.textContent = snapshot.student_count;
  avgAttentionEl.textContent = `${snapshot.average_attention.toFixed(1)}%`;
  lowAttentionEl.textContent = snapshot.low_attention_students;

  const when = new Date(snapshot.last_updated);
  lastUpdatedEl.textContent = `Last update: ${when.toLocaleTimeString()}`;

  rowsEl.innerHTML = snapshot.students
    .map((s) => {
      const cls = `badge-${s.attention_band}`;
      return `
        <tr>
          <td>${esc(s.student_id)}</td>
          <td>${s.attention_score.toFixed(1)}</td>
          <td><span class="badge ${cls}">${esc(s.attention_band)}</span></td>
          <td>${(s.gaze_focus * 100).toFixed(0)}%</td>
          <td>${(s.head_motion * 100).toFixed(0)}%</td>
          <td>${(s.ambient_noise * 100).toFixed(0)}%</td>
          <td>${s.blink_rate.toFixed(1)}/min</td>
        </tr>
      `;
    })
    .join("");
}

async function bootstrap() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${window.location.host}/ws/classroom`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    render(data);
  };

  ws.onopen = () => {
    setInterval(() => ws.send("ping"), 15000);
  };

  ws.onerror = async () => {
    const snapshot = await fetch("/api/snapshot").then((r) => r.json());
    render(snapshot);
  };
}

bootstrap();
