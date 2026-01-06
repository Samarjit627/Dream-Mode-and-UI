import { Router } from 'express'
import analyze from './analyze.js'
import ideate from './ideate.js'
import mentor from './mentor.js'
import taste from './taste.js'
import knowledge from './knowledge.js'
import health from './health.js'
import execRoutes from './exec.js'
import upload from './upload.js'
import dream from './dream.js'
import chat from './chat.js'

const r = Router()

r.use(dream)
r.use(chat)
r.use(analyze)
r.use(ideate)
r.use(mentor)
r.use(taste)
r.use(knowledge)
r.use(execRoutes)
r.use(upload)
r.use(health)

export default r
