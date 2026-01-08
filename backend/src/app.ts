import express from 'express'
import cors from 'cors'
import pinoHttp from 'pino-http'
import 'dotenv/config'
import routes from './routes/index.js'

const app = express()
app.use(cors())
app.use(express.json({ limit: '50mb' }))
app.use(pinoHttp())

app.use('/api', routes)

export default app
