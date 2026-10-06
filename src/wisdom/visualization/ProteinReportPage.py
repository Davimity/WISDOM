"""Portable protein galleries for LambdaForge's isolated project-report sections."""

# ruff: noqa: E501 -- browser-native presentation remains a readable embedded template.

from __future__ import annotations

import gzip
import json
import base64

from typing import Any
from collections import defaultdict
from plotly.offline import get_plotlyjs
from collections.abc import Mapping, Sequence


class ProteinReportPage:
    """Package existing inspectors with one shared library and a bounded offline navigator."""

    def __init__(self, maximum_mib: float = 4.0) -> None:
        """Set the presentation budget without changing scientific evaluation coverage.

        Args:
            maximum_mib: Maximum UTF-8 document size in MiB, positive and at most LambdaForge's
                16 MiB/document limit. The framework separately enforces 64 MiB/report.

        Raises:
            ValueError: If the budget exceeds the framework's document limit or is non-positive.
        """
        if not 0.0 < maximum_mib <= 16.0:
            raise ValueError("protein report maximum_mib must lie in (0,16]")
        self.maximum_bytes = int(maximum_mib * 1024 * 1024)

    @staticmethod
    def _encode(text: str) -> str:
        """Encode deterministic gzip data for a modern browser's DecompressionStream.

        Args:
            text: UTF-8 document or JavaScript owned by the shared WISDOM renderer.

        Returns:
            ASCII base64 containing gzip bytes; no files or network requests are needed.
        """
        return base64.b64encode(gzip.compress(text.encode("utf-8"), mtime=0)).decode("ascii")

    def render(
        self,
        documents: Sequence[Mapping[str, Any]],
        context  : Mapping[str, Any],
        reason   : str = "No protein visualizations were generated for this Run.",
    ) -> tuple[str, dict[str, Any]]:
        """Build a searchable gallery reusing exactly the preprocessing inspector.

        Documents contain ``identifier``, ``split``, ``label`` and ``html``. The HTML must come
        from ProteinVisualizer.render(plotly_script=False), so the library is packaged only once.
        A deterministic round-robin over model/seed/split/label strata bounds presentation without
        preferring one class or checkpoint. Omitted viewers remain explicit; metrics are not filtered.

        Args:
            documents: Generated, already point-aligned inspectors; no NPZ is read here.
            context: JSON-compatible Run/checkpoint metadata displayed without HTML interpolation.
            reason: Empty-state explanation, including disabled or pruned visualization.

        Returns:
            Self-contained HTML and a JSON-compatible included/omitted/byte-count summary.

        Raises:
            ValueError: If a document lacks its renderer-owned HTML or the shell cannot fit.
        """
        # One compressed Plotly bundle serves whichever protein is currently open. Data and
        # templates are decompressed lazily; no nested iframe or relative asset escapes LF's CSP.

        library = self._encode(get_plotlyjs()) if documents else ""
        buckets : dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
        for document in documents:
            buckets[(str(document.get("variant", "")), str(document.get("seed", "")),
                     str(document["split"]), int(document["label"]))].append(document)
        for bucket in buckets.values():
            bucket.sort(key=lambda item: str(item["identifier"]))

        entries : list[dict[str, Any]] = []
        omitted : list[str]           = []
        used                          = len(library) + 32 * 1024
        row                           = 0
        while any(row < len(bucket) for bucket in buckets.values()):
            for key in sorted(buckets):
                bucket = buckets[key]
                if row >= len(bucket):
                    continue
                document = bucket[row]
                encoded  = self._encode(str(document["html"]))
                entry    = {
                    "identifier": str(document["identifier"]),
                    "split":      str(document["split"]),
                    "label":      int(document["label"]),
                    "content":    encoded,
                    "variant":    str(document.get("variant", "")),
                    "seed":       document.get("seed"),
                    "details":    document.get("details", {}),
                }
                size = len(json.dumps(entry).encode("utf-8"))
                if used + size <= self.maximum_bytes:
                    entries.append(entry)
                    used += size
                else:
                    omitted.append(f"{entry['variant']} seed={entry['seed']} {entry['split']}/{entry['identifier']}")
            row += 1
        if not entries:
            library = ""
            if documents:
                reason = "The configured report byte budget cannot fit a protein viewer. Use visualization.content=predictions, reduce visualization.maximum_points, or increase visualization.report_budget_mib."

        payload = json.dumps({
            "documents": entries,
            "plotly":    library,
            "context":   dict(context),
            "omitted":   omitted,
            "reason":    reason,
        }, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
        page = self._html(payload)
        if len(page.encode("utf-8")) > self.maximum_bytes:
            raise ValueError("protein report shell exceeds its configured byte budget")
        return page, {
            "included_proteins": len(entries),
            "omitted_proteins":  omitted,
            "size_bytes":        len(page.encode("utf-8")),
        }

    @staticmethod
    def _html(payload: str) -> str:
        """Compose local navigation around trusted shared-renderer documents.

        Args:
            payload: Script-safe JSON with compressed library/documents and escaped-as-text context.

        Returns:
            Complete HTML; only the active protein owns a WebGL scene. Returning to the list waits
            for pending updates, purges Plotly, and removes the inspector's resize listener.
        """
        template = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>WISDOM proteins</title>
<style id="wisdom-gallery-style">
body.wisdom-gallery{margin:0;padding:28px;background:#080c16;color:#e7edf7;font:14px system-ui,sans-serif;display:block;height:auto;overflow:auto}
.wisdom-gallery h1{margin:0 0 12px}.wisdom-gallery p{line-height:1.6;color:#a9b7ca}.wisdom-gallery input,.wisdom-gallery select,.wisdom-gallery button{padding:10px;margin:4px;background:#172236;color:#e7edf7;border:1px solid #3b4a61;border-radius:8px}
.wisdom-gallery table{width:100%;border-collapse:collapse;margin-top:18px}.wisdom-gallery td,.wisdom-gallery th{text-align:left;padding:12px;border-bottom:1px solid #27364b}.wisdom-gallery pre{white-space:pre-wrap;color:#a9b7ca}.wisdom-gallery [hidden]{display:none}.wisdom-back{margin-right:8px}.wisdom-gallery .notice{border-left:3px solid #fdb022;padding:12px;background:#2d2415}
</style></head><body class="wisdom-gallery"><p>Loading WISDOM gallery…</p>
<script id="wisdom-gallery-data" type="application/json">@@PAYLOAD@@</script><script>
(()=>{
const data=JSON.parse(document.getElementById('wisdom-gallery-data').textContent);
const filter={query:'',split:'',label:'',variant:'',seed:''};let busy=false,libraryReady;
async function decode(encoded){if(!window.DecompressionStream)throw Error('This viewer requires a modern browser with DecompressionStream (Chrome, Firefox or Safari).');const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text()}
function element(tag,text){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;return node}
async function closeViewer(){if(window.wisdomDispose){await window.wisdomDispose();window.wisdomDispose=null}document.getElementById('wisdom-protein-style')?.remove();document.body.className='wisdom-gallery';document.body.replaceChildren()}
async function catalogue(message=''){
await closeViewer();document.body.append(element('h1','WISDOM proteins'));
document.body.append(element('p','Choose a source Run in LambdaForge or filter reviewed models and seeds below, then open a protein. This gallery never selects a scientific winner.'));
document.body.append(element('pre',JSON.stringify(data.context,null,2)));
if(message)document.body.append(element('p',message));
if(data.omitted.length){const notice=element('p',`${data.omitted.length} viewers omitted to respect the byte budget: ${data.omitted.join(', ')}. Evaluation metrics still cover the complete evaluated split.`);notice.className='notice';document.body.append(notice)}
if(!data.documents.length){document.body.append(element('p',data.reason));return}
const controls=element('div'),search=element('input');search.type='search';search.placeholder='Search protein ID';search.value=filter.query;search.setAttribute('aria-label','Search protein');
const split=element('select'),label=element('select'),variant=element('select');split.setAttribute('aria-label','Split');label.setAttribute('aria-label','Protein label');variant.setAttribute('aria-label','Model or pooling');
const seed=element('select');seed.setAttribute('aria-label','Source seed');for(const value of ['',...new Set(data.documents.map(d=>String(d.seed??'unseeded')))]){const option=element('option',value?`Seed ${value}`:'All source seeds');option.value=value;seed.append(option)}seed.value=filter.seed;
for(const value of ['',...new Set(data.documents.map(d=>d.variant).filter(Boolean))]){const option=element('option',value||'All models / poolings');option.value=value;variant.append(option)}variant.value=filter.variant;
for(const value of ['',...new Set(data.documents.map(d=>d.split))]){const option=element('option',value||'All splits');option.value=value;split.append(option)}
for(const [value,title] of [['','Both labels'],['0','Negative'],['1','Positive']]){const option=element('option',title);option.value=value;label.append(option)}split.value=filter.split;label.value=filter.label;controls.append(search,variant,seed,split,label);document.body.append(controls);
const table=element('table'),header=element('tr');for(const name of ['Model / pooling','Seed','Protein','Split','Label','Case diagnostics','Viewer'])header.append(element('th',name));table.append(header);document.body.append(table);
const rows=data.documents.map((entry,index)=>{const row=element('tr');row.dataset.protein=entry.identifier;for(const text of [entry.variant||'Source model',entry.seed??'—',entry.identifier,entry.split,entry.label?'Positive':'Negative'])row.append(element('td',text));const details=element('td');details.append(element('pre',Object.keys(entry.details).length?JSON.stringify(entry.details,null,2):'—'));row.append(details);const cell=element('td'),button=element('button','Open protein');button.addEventListener('click',()=>open(index));cell.append(button);row.append(cell);table.append(row);return row});
const update=()=>{filter.query=search.value;filter.split=split.value;filter.label=label.value;filter.variant=variant.value;filter.seed=seed.value;rows.forEach((row,index)=>{const d=data.documents[index];row.hidden=Boolean(!d.identifier.toLowerCase().includes(filter.query.trim().toLowerCase())||(filter.split&&d.split!==filter.split)||(filter.label&&String(d.label)!==filter.label)||(filter.variant&&d.variant!==filter.variant)||(filter.seed&&String(d.seed??'unseeded')!==filter.seed))})};search.addEventListener('input',update);split.addEventListener('change',update);label.addEventListener('change',update);variant.addEventListener('change',update);seed.addEventListener('change',update);update();
}
async function open(index){
if(busy)return;busy=true;
try{
document.body.append(element('p','Loading protein and WebGL…'));
if(!libraryReady)libraryReady=decode(data.plotly).then(source=>{new Function(source)()});await libraryReady;
const source=await decode(data.documents[index].content),parsed=new DOMParser().parseFromString(source,'text/html');
await closeViewer();const style=element('style');style.id='wisdom-protein-style';style.textContent=[...parsed.querySelectorAll('style')].map(node=>node.textContent).join('\n');document.head.append(style);
const scripts=[...parsed.body.querySelectorAll('script')].map(node=>{if(node.src)throw Error('Protein documents must not load external scripts.');const source=node.textContent;node.remove();return source});
document.body.className='';document.body.innerHTML=parsed.body.innerHTML;window.wisdomPlotReady=null;window.wisdomViewerReady=null;
for(const script of scripts)new Function(script)();await window.wisdomViewerReady;
const back=element('button','← Protein list');back.className='wisdom-back';back.addEventListener('click',async()=>{if(busy)return;busy=true;try{await catalogue()}catch(error){console.error(error)}finally{busy=false}});document.querySelector('.viewer-head').prepend(back);const identity=element('span',`${data.documents[index].variant||'Source model'} · seed ${data.documents[index].seed??'—'}`);document.querySelector('.viewer-head').append(identity);
}catch(error){console.error(error);await catalogue(`Viewer unavailable: ${error.message}`)}finally{busy=false}
}
catalogue().catch(error=>{document.body.textContent=`Gallery unavailable: ${error.message}`});
})();
</script></body></html>'''
        return template.replace("@@PAYLOAD@@", payload)
