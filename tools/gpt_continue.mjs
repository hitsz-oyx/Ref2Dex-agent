// Append one message to an existing ChatGPT conversation. A check does not send.
const CDP = 'http://127.0.0.1:9222'
const target = process.argv[2] || ''
const check = process.argv.includes('--check')
const msgPath = process.argv[3]
if (!/\/c\/[0-9a-f-]+/i.test(target)) {
  console.log('BAD_URL')
  process.exit(1)
}
const id = target.match(/\/c\/([0-9a-f-]+)/i)[1]
const url = target.startsWith('http') ? target : `https://chatgpt.com/c/${id}`
const sleep = ms => new Promise(r => setTimeout(r, ms))
const fresh = await (await fetch(`${CDP}/json/new?${encodeURIComponent(url)}`, { method: 'PUT' })).json()
const ws = new WebSocket(fresh.webSocketDebuggerUrl)
let seq = 0
const pending = new Map()
ws.addEventListener('message', ev => {
  let msg
  try { msg = JSON.parse(ev.data) } catch { return }
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id) }
})
const call = (method, params = {}, to = 30000) => {
  const i = ++seq
  ws.send(JSON.stringify({ id: i, method, params }))
  return new Promise((res, rej) => {
    const t = setTimeout(() => rej(new Error('timeout ' + method)), to)
    pending.set(i, msg => { clearTimeout(t); msg.error ? rej(new Error(JSON.stringify(msg.error))) : res(msg.result) })
  })
}
const js = async (expr) => {
  const result = await call('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })
  return result.result?.value
}
const clickAt = async (x, y) => {
  await call('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y })
  await sleep(150)
  await call('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', clickCount: 1 })
  await call('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', clickCount: 1 })
}
await new Promise(r => ws.addEventListener('open', r))
await call('Runtime.enable')
await sleep(8000)
const here = String(await js('location.href'))
if (!here.includes(id)) {
  console.log('WRONG_PAGE', here)
  ws.close()
  process.exit(1)
}
const box = JSON.parse(await js(`(() => {
  const el = document.querySelector('#prompt-textarea') || document.querySelector('div[contenteditable="true"]');
  if (!el) return 'null';
  const r = el.getBoundingClientRect();
  return JSON.stringify({ x: Math.round(r.x + r.width / 2), y: Math.round(r.y + r.height / 2) });
})()`) || 'null')
if (!box) {
  console.log('NO_COMPOSER')
  ws.close()
  process.exit(1)
}
console.log('COMPOSER_OK', id)
if (check) {
  ws.close()
  process.exit(0)
}
const fs = await import('node:fs')
const text = fs.readFileSync(msgPath, 'utf8')
await clickAt(box.x, box.y)
await sleep(300)
await call('Input.insertText', { text })
await sleep(800)
const send = JSON.parse(await js(`(() => {
  const nodes = [...document.querySelectorAll('button')].filter(b => /发送|Send/i.test((b.getAttribute('aria-label') || '') + (b.innerText || '')) && b.getBoundingClientRect().width > 0);
  const b = nodes[nodes.length - 1];
  if (!b) return 'null';
  const r = b.getBoundingClientRect();
  return JSON.stringify({ x: Math.round(r.x + r.width / 2), y: Math.round(r.y + r.height / 2) });
})()`) || 'null')
if (!send) {
  console.log('NO_SEND')
  ws.close()
  process.exit(1)
}
await clickAt(send.x, send.y)
await sleep(2000)
const after = String(await js('location.href'))
console.log('SENT', after.includes(id) ? id : 'LEFT')
ws.close()
if (!after.includes(id)) process.exit(1)
