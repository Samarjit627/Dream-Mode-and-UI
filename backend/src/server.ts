import { createServer } from 'http'
import app from './app.js'

const PORT = process.env.PORT ? Number(process.env.PORT) : 7071
const server = createServer(app)

server.listen(PORT, () => {
  console.log(`[gateway] listening on http://localhost:${PORT}`)
})
