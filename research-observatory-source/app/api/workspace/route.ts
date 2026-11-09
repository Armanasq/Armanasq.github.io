import { env } from 'cloudflare:workers';
export const dynamic = 'force-dynamic';
const reply=(data:unknown,status=200)=>Response.json(data,{status,headers:{'Cache-Control':'private, no-store'}});
export async function GET(req:Request){
 if(!env.DB)return reply({error:'Workspace storage unavailable'},503);
 const owner=req.headers.get('oai-authenticated-user-id');
 if(!owner)return reply({error:'Sign in to save and load your workspace.'},401);
 const row=await env.DB.prepare('SELECT payload, revision, updated FROM workspaces WHERE owner = ?').bind(owner).first();
 return reply(row?{data:JSON.parse(String(row.payload)),revision:row.revision,updated:row.updated}:{data:{notes:{},runs:[],synthesis:''},revision:0});
}
export async function POST(req:Request){
 if(!env.DB)return reply({error:'Workspace storage unavailable'},503);
 const owner=req.headers.get('oai-authenticated-user-id');
 if(!owner)return reply({error:'Sign in required'},401);
 if(req.headers.get('origin')!==new URL(req.url).origin)return reply({error:'Origin rejected'},403);
 const raw=await req.text();if(raw.length>8000000)return reply({error:'Workspace exceeds 8 MB'},413);
 let body;try{body=JSON.parse(raw)}catch{return reply({error:'Invalid JSON'},400)}
 if(!Number.isInteger(body.revision)||body.revision<0||!body.data||!Array.isArray(body.data.runs)||body.data.runs.length>3000||!body.data.notes||typeof body.data.notes!=='object'||typeof body.data.synthesis!=='string')return reply({error:'Invalid workspace schema'},400);
 const ids=new Set();for(const r of body.data.runs){if(!r||typeof r.run_id!=='string'||!r.run_id.trim()||typeof r.scenario_id!=='string'||typeof r.variant!=='string'||!Array.isArray(r.turns)||ids.has(r.run_id))return reply({error:'Invalid or duplicate run'},400);ids.add(r.run_id);for(const k of ['latency_ms','total_tokens','cost_usd'])if(r[k]!=null&&(typeof r[k]!=='number'||!Number.isFinite(r[k])||r[k]<0))return reply({error:'Invalid metric'},400);}
 const payload=JSON.stringify(body.data),updated=new Date().toISOString();
 const result=body.revision===0?await env.DB.prepare('INSERT OR IGNORE INTO workspaces (owner,payload,revision,updated) VALUES (?,?,1,?)').bind(owner,payload,updated).run():await env.DB.prepare('UPDATE workspaces SET payload=?,revision=revision+1,updated=? WHERE owner=? AND revision=?').bind(payload,updated,owner,body.revision).run();
 if(!result.meta.changes)return reply({error:'Another tab changed this workspace. Export edits, then reload before saving.'},409);
 return reply({revision:body.revision+1,updated});
}
