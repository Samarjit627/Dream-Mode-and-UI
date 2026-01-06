import { createServer } from 'http'
import app from './app.js'

const PORT = process.env.PORT ? Number(process.env.PORT) : 7072
const server = createServer(app)

server.listen(PORT, () => {
  console.log(`[worker] listening on http://localhost:${PORT}`)
})
