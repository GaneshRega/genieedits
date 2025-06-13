// resume_builder.js
// Smart Resume Generator JS logic

// --- Autosave/Restore ---
function saveToLocal(key, value) {
    localStorage.setItem(key, JSON.stringify(value));
}
function loadFromLocal(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) || fallback; } catch { return fallback; }
}

// --- Dynamic Field Management ---
function addEntry(listId, fields, values = {}) {
    const list = document.getElementById(listId);
    const entry = document.createElement('div');
    entry.className = 'entry';
    fields.forEach(f => {
        let el;
        if (f.type === 'textarea') {
            el = document.createElement('textarea');
            el.rows = 1;
        } else {
            el = document.createElement('input');
            el.type = f.type;
        }
        el.placeholder = f.placeholder;
        el.value = values[f.name] || '';
        el.name = f.name;
        el.addEventListener('input', updatePreview); // Ensure live preview on input
        entry.appendChild(el);
    });
    const removeBtn = document.createElement('button');
    removeBtn.className = 'remove-btn';
    removeBtn.type = 'button';
    removeBtn.innerHTML = '×';
    removeBtn.onclick = () => { entry.remove(); updatePreview(); };
    entry.appendChild(removeBtn);
    list.appendChild(entry);
    updatePreview(); // Update preview after adding entry
}

// --- Skills Tag Input ---
const skillSuggestions = ['Python','JavaScript','HTML','CSS','Flask','Django','React','SQL','Machine Learning','Data Analysis','Git','C++','Java','AWS','Docker'];
let skills = loadFromLocal('skills', []);
function renderSkills() {
    const container = document.getElementById('skills-input');
    container.innerHTML = '';
    skills.forEach((skill, i) => {
        const tag = document.createElement('span');
        tag.className = 'tag';
        tag.textContent = skill;
        const rm = document.createElement('span');
        rm.className = 'remove-tag';
        rm.textContent = '×';
        rm.onclick = () => { skills.splice(i,1); saveToLocal('skills',skills); renderSkills(); updatePreview(); };
        tag.appendChild(rm);
        container.appendChild(tag);
    });
}
document.getElementById('skill-suggest').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && this.value.trim()) {
        e.preventDefault();
        if (!skills.includes(this.value.trim())) {
            skills.push(this.value.trim());
            saveToLocal('skills',skills);
            renderSkills();
            updatePreview();
        }
        this.value = '';
    }
});
renderSkills();

// --- Add/Remove Dynamic Fields ---
const eduFields = [
    {name:'degree',type:'text',placeholder:'Degree'},
    {name:'year',type:'text',placeholder:'Year'},
    {name:'institute',type:'text',placeholder:'Institute'}
];
const expFields = [
    {name:'title',type:'text',placeholder:'Title'},
    {name:'company',type:'text',placeholder:'Company'},
    {name:'year',type:'text',placeholder:'Year'},
    {name:'desc',type:'textarea',placeholder:'Description'}
];
const projFields = [
    {name:'title',type:'text',placeholder:'Title'},
    {name:'desc',type:'textarea',placeholder:'Description'},
    {name:'link',type:'text',placeholder:'Link'}
];

document.getElementById('add-education').onclick = () => addEntry('education-list', eduFields);
document.getElementById('add-experience').onclick = () => addEntry('experience-list', expFields);
document.getElementById('add-project').onclick = () => addEntry('project-list', projFields);

// --- Template Selection ---
let selectedTemplate = 'modern';
document.querySelectorAll('.template-thumb').forEach(card => {
    card.addEventListener('click', function() {
        document.querySelectorAll('.template-thumb').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        selectedTemplate = card.getAttribute('data-template');
        updatePreview();
    });
});

// --- Form Autosave/Restore ---
const form = document.getElementById('resume-form');
form.oninput = function() {
    const data = Object.fromEntries(new FormData(form).entries());
    saveToLocal('resumeForm', data);
    updatePreview(); // Ensure preview updates on every input
};
window.onload = function() {
    // Restore form
    const data = loadFromLocal('resumeForm', {});
    for (const k in data) if (form[k]) form[k].value = data[k];
    // Restore dynamic fields
    (loadFromLocal('education',[])).forEach(e => addEntry('education-list', eduFields, e));
    (loadFromLocal('experience',[])).forEach(e => addEntry('experience-list', expFields, e));
    (loadFromLocal('projects',[])).forEach(e => addEntry('project-list', projFields, e));
    showStep(1);
    updatePreview();
};

// --- Multi-step navigation logic ---
const steps = [1,2,3,4];
let currentStep = 1;
function showStep(step) {
    steps.forEach(i => {
        document.getElementById('step-'+i).style.display = (i===step)?'block':'none';
        document.getElementById('step-bar-'+i).classList.toggle('active',i<=step);
    });
    currentStep = step;
    updatePreview();
}
document.getElementById('to-step-2').onclick = function() { showStep(2); };
document.getElementById('to-step-3').onclick = function() { showStep(3); };
document.getElementById('to-step-4').onclick = function() { showStep(4); };
document.getElementById('back-to-1').onclick = function() { showStep(1); };
document.getElementById('back-to-2').onclick = function() { showStep(2); };
document.getElementById('back-to-3').onclick = function() { showStep(3); };

// --- Live Preview ---
function updatePreview() {
    const data = Object.fromEntries(new FormData(form).entries());
    data.skills = skills;
    data.education = Array.from(document.querySelectorAll('#education-list .entry')).map(e => {
        const [degree, year, institute] = e.querySelectorAll('input');
        return {degree:degree.value, year:year.value, institute:institute.value};
    });
    data.experience = Array.from(document.querySelectorAll('#experience-list .entry')).map(e => {
        const [title, company, year, desc] = e.querySelectorAll('input,textarea');
        return {title:title.value, company:company.value, year:year.value, desc:desc.value};
    });
    data.projects = Array.from(document.querySelectorAll('#project-list .entry')).map(e => {
        const [title, desc, link] = e.querySelectorAll('input,textarea');
        return {title:title.value, desc:desc.value, link:link.value};
    });
    saveToLocal('education', data.education);
    saveToLocal('experience', data.experience);
    saveToLocal('projects', data.projects);
    // Render main preview
    document.getElementById('resume-preview').innerHTML = renderPreview(data, selectedTemplate);
    // Render simplified PDF preview (no gradients, no border-radius, no box-shadow)
    document.getElementById('resume-preview-pdf').innerHTML = renderPreviewPDF(data, selectedTemplate);
}

// --- Main Preview Renderer ---
function renderPreview(data, template) {
    if (!data || Object.keys(data).length === 0) {
        return '<div style="color:#aaa;text-align:center;padding:2em 0;">Your resume preview will appear here as you fill the form.</div>';
    }
    let html = `
    <div style="background:#fff;padding:32px 24px;max-width:800px;width:800px;margin:auto;font-family:'Poppins','Inter',Arial,sans-serif;color:#222;">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:24px;min-height:40px;">
            <img src='/static/logo.png' alt='Logo' style='height:40px;width:40px;object-fit:contain;display:inline-block;vertical-align:middle;border-radius:8px;background:#f5f5f5;' onerror='this.style.display="none";'>
            <span style='font-size:1.5em;font-weight:800;color:#7B61FF;letter-spacing:1px;display:inline-block;vertical-align:middle;'>Smart Resume Generator</span>
        </div>
        <div style="text-align:center;margin-bottom:18px;">
            <div style="font-size:2.2em;font-weight:800;color:#222;letter-spacing:1px;">${data.name||form.name.value||'Your Name'}</div>
            <div style="font-size:1.1em;font-weight:600;color:#444;margin-bottom:0.7em;">${data.summary||form.summary.value||'Your Job Title / Career Objective'}</div>
        </div>
        <div style="margin:2em 0 0 0;">
            <div style="margin-bottom:1.2em;"><b>Contact:</b> ${(data.email||form.email.value||'')} | ${(data.phone||form.phone.value||'')}</div>
            <div style="margin-bottom:1.2em;"><b>Education:</b> ${listSection(data.education,'degree, institute, year')}</div>
            <div style="margin-bottom:1.2em;"><b>Certifications:</b> ${data.certifications||form.certifications.value||''}</div>
            <div style="margin-bottom:1.2em;"><b>Experience:</b> ${listSection(data.experience,'title, company, year, desc')}</div>
            <div style="margin-bottom:1.2em;"><b>Projects:</b> ${listSection(data.projects,'title, desc, link')}</div>
            <div style="margin-bottom:1.2em;"><b>Skills:</b> ${tagSection(data.skills)}</div>
            <div style="margin-bottom:1.2em;"><b>Languages:</b> ${data.languages||form.languages.value||''}</div>
        </div>
    </div>`;
    return html;
}

// --- PDF-friendly preview renderer ---
function renderPreviewPDF(data, template) {
    let html = '';
    html = `
    <div style="background:#fff;padding:32px 24px;max-width:800px;width:800px;margin:auto;font-family:'Poppins','Inter',Arial,sans-serif;color:#222;">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:24px;min-height:40px;">
            <img src='/static/logo.png' alt='Logo' style='height:40px;width:40px;object-fit:contain;display:inline-block;vertical-align:middle;border-radius:8px;background:#f5f5f5;' onerror='this.style.display=\"none\";'>
            <span style='font-size:1.5em;font-weight:800;color:#7B61FF;letter-spacing:1px;display:inline-block;vertical-align:middle;'>Smart Resume Generator</span>
        </div>
        <div style="text-align:center;margin-bottom:18px;">
            <div style="font-size:2.2em;font-weight:800;color:#222;letter-spacing:1px;">${data.name||form.name.value||'Your Name'}</div>
            <div style="font-size:1.1em;font-weight:600;color:#444;margin-bottom:0.7em;">${data.summary||form.summary.value||'Your Job Title / Career Objective'}</div>
        </div>
        <div style="margin:2em 0 0 0;">
            <div style="margin-bottom:1.2em;"><b>Contact:</b> ${(data.email||form.email.value||'')} | ${(data.phone||form.phone.value||'')}</div>
            <div style="margin-bottom:1.2em;"><b>Education:</b> ${listSection(data.education,'degree, institute, year')}</div>
            <div style="margin-bottom:1.2em;"><b>Certifications:</b> ${data.certifications||form.certifications.value||''}</div>
            <div style="margin-bottom:1.2em;"><b>Experience:</b> ${listSection(data.experience,'title, company, year, desc')}</div>
            <div style="margin-bottom:1.2em;"><b>Projects:</b> ${listSection(data.projects,'title, desc, link')}</div>
            <div style="margin-bottom:1.2em;"><b>Skills:</b> ${tagSection(data.skills)}</div>
            <div style="margin-bottom:1.2em;"><b>Languages:</b> ${data.languages||form.languages.value||''}</div>
        </div>
    </div>`;
    return html;
}
