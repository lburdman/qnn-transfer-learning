// Organizador de eventos - Vanilla JS (sin frameworks)
(function () {
  'use strict';

  // --------------------- Config y constantes ---------------------
  const TZ = 'America/Argentina/Buenos_Aires';
  const LS_SPEAKERS = 'qff.speakers';
  const LS_EVENTS = 'qff.events';
  const FRANJAS = ['Mañana', 'Tarde', 'Noche'];
  const NIVELES = ['Bajo', 'Intermedio', 'Alto'];

  // --------------------- Estado ---------------------
  const state = {
    speakers: [],
    events: [],
    filters: { speakerId: '', tematica: '', nivel: '' },
    sortSpeakers: { key: 'nombre', dir: 'asc' },
    startISO: null, // lunes de la primera semana visible
    config: { allowStack: true },
  };

  // --------------------- Utilidades de fecha ---------------------
  function pad2(n) { return String(n).padStart(2, '0'); }

  function getTodayISOInBA() {
    const now = new Date();
    const parts = new Intl.DateTimeFormat('en-CA', {
      timeZone: TZ,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    }).formatToParts(now);
    const by = Object.fromEntries(parts.map(p => [p.type, p.value]));
    return `${by.year}-${by.month}-${by.day}`;
  }

  function parseISO(iso) {
    const [y, m, d] = iso.split('-').map(Number);
    return { y, m, d };
  }

  function toISOFromUTCDate(dt) {
    const y = dt.getUTCFullYear();
    const m = pad2(dt.getUTCMonth() + 1);
    const d = pad2(dt.getUTCDate());
    return `${y}-${m}-${d}`;
  }

  function addDaysISO(iso, delta) {
    const { y, m, d } = parseISO(iso);
    const t = Date.UTC(y, m - 1, d) + delta * 86400000;
    return toISOFromUTCDate(new Date(t));
  }

  // Sakamoto (0=Dom..6=Sab). Convertimos a 0=Lun..6=Dom
  function weekdayMonday0(iso) {
    const { y, m, d } = parseISO(iso);
    const t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let Y = y;
    let M = m;
    if (M < 3) Y -= 1;
    const wSun0 = (Y + Math.floor(Y / 4) - Math.floor(Y / 100) + Math.floor(Y / 400) + t[M - 1] + d) % 7;
    // wSun0: 0=Dom..6=Sab; queremos 0=Lun..6=Dom
    const wMon0 = (wSun0 + 6) % 7;
    return wMon0;
  }

  function adjustToMonday(iso) {
    const w = weekdayMonday0(iso); // 0=Lun..6=Dom
    return addDaysISO(iso, -w);
  }

  function buildFourWeekRange(startISO) {
    // Asume startISO es lunes
    const days = [];
    for (let i = 0; i < 28; i++) days.push(addDaysISO(startISO, i));
    return days;
  }

  function formatISO_DDMMYYYY(iso) {
    const { y, m, d } = parseISO(iso);
    return `${pad2(d)}/${pad2(m)}/${y}`;
  }

  function weekdayShortEs(wMon0) {
    // 0=Lun..6=Dom
    return ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'][wMon0] || '';
  }

  function franjaOrder(f) {
    return { 'Mañana': 0, 'Tarde': 1, 'Noche': 2 }[f] ?? 99;
  }

  // --------------------- Persistencia ---------------------
  function saveSpeakersToLS(speakers) {
    localStorage.setItem(LS_SPEAKERS, JSON.stringify(speakers));
  }
  function saveEventsToLS(events) {
    localStorage.setItem(LS_EVENTS, JSON.stringify(events));
  }
  function loadFromLS() {
    const spRaw = localStorage.getItem(LS_SPEAKERS);
    const evRaw = localStorage.getItem(LS_EVENTS);
    let speakers = [];
    let events = [];
    if (spRaw) {
      try { speakers = JSON.parse(spRaw) || []; } catch { speakers = []; }
    }
    if (evRaw) {
      try { events = JSON.parse(evRaw) || []; } catch { events = []; }
    }
    return { speakers, events };
  }

  function seedIfEmpty() {
    const { speakers, events } = loadFromLS();
    if (speakers.length === 0 && events.length === 0) {
      const spk1 = { id: 'spk_ana', nombre: 'Ana Pérez', tematica: 'IA aplicada', bio: 'Ingeniera de datos. Experiencia en ML en producción.' };
      const spk2 = { id: 'spk_carlos', nombre: 'Carlos Gómez', tematica: 'DevOps', bio: 'Plataformas y SRE, Kubernetes y GitOps.' };
      const spk3 = { id: 'spk_lucia', nombre: 'Lucía Fernández', tematica: 'Frontend', bio: 'UI accesible y rendimiento en la web.' };
      const today = getTodayISOInBA();
      const monday = adjustToMonday(today);
      const d1 = addDaysISO(monday, 1); // Mar
      const d2 = addDaysISO(monday, 3); // Jue
      const d3 = addDaysISO(monday, 8); // Semana 2, Mar
      const evt1 = {
        id: 'evt_1', fecha: d1, franja: 'Mañana', speakers: [spk1.id], tematica: 'Transformers 101', descripcion: 'Intro breve', nivel: 'Intermedio'
      };
      const evt2 = {
        id: 'evt_2', fecha: d2, franja: 'Noche', speakers: [spk2.id], tematica: 'CI/CD moderno', descripcion: 'Pipelines y seguridad', nivel: 'Alto'
      };
      const evt3 = {
        id: 'evt_3', fecha: d3, franja: 'Tarde', speakers: [spk3.id], tematica: 'Accesibilidad web', descripcion: 'Buenas prácticas', nivel: 'Bajo'
      };
      saveSpeakersToLS([spk1, spk2, spk3]);
      saveEventsToLS([evt1, evt2, evt3]);
    }
  }

  // --------------------- Utilidades varias ---------------------
  function generateId(prefix) {
    const rnd = Math.random().toString(36).slice(2, 8);
    return `${prefix}_${Date.now().toString(36)}_${rnd}`;
  }

  function getSpeakerById(id) {
    return state.speakers.find(s => s.id === id);
  }

  function speakersToOptions(selectEl, selectedIds = []) {
    selectEl.innerHTML = '';
    state.speakers.forEach(sp => {
      const opt = document.createElement('option');
      opt.value = sp.id;
      opt.textContent = sp.nombre;
      if (selectedIds.includes(sp.id)) opt.selected = true;
      selectEl.appendChild(opt);
    });
  }

  function showToast(msg, type = 'success') {
    const cont = document.getElementById('toastContainer');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = msg;
    cont.appendChild(el);
    setTimeout(() => { el.remove(); }, 3000);
  }

  function matchesFiltersSpeaker(sp) {
    const { speakerId, tematica } = state.filters;
    if (speakerId && sp.id !== speakerId) return false;
    if (tematica && !(sp.tematica || '').toLowerCase().includes(tematica.toLowerCase())) return false;
    return true;
  }

  function matchesFiltersEvent(ev) {
    const { speakerId, tematica, nivel } = state.filters;
    if (speakerId && !ev.speakers.includes(speakerId)) return false;
    if (nivel && ev.nivel !== nivel) return false;
    if (tematica && !(ev.tematica || '').toLowerCase().includes(tematica.toLowerCase())) return false;
    return true;
  }

  function sortEventsByDateSlot(a, b) {
    if (a.fecha !== b.fecha) return a.fecha < b.fecha ? -1 : 1;
    return franjaOrder(a.franja) - franjaOrder(b.franja);
  }

  function checkSlotOccupied(fechaISO, franja, ignoreEventId = null) {
    return state.events.filter(e => e.fecha === fechaISO && e.franja === franja && e.id !== ignoreEventId);
  }

  function uniqueLevels(events) {
    return Array.from(new Set(events.map(e => e.nivel)));
  }

  // --------------------- Render: Speakers ---------------------
  function renderSpeakersTable() {
    const body = document.getElementById('speakersTableBody');
    body.innerHTML = '';
    const filtered = state.speakers.filter(matchesFiltersSpeaker);
    const { key, dir } = state.sortSpeakers;
    filtered.sort((a, b) => {
      const av = (a[key] || '').toLowerCase();
      const bv = (b[key] || '').toLowerCase();
      if (av < bv) return dir === 'asc' ? -1 : 1;
      if (av > bv) return dir === 'asc' ? 1 : -1;
      return 0;
    });
    for (const sp of filtered) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${sp.nombre}</td><td>${sp.tematica || ''}</td><td>${sp.bio || ''}</td>`;
      body.appendChild(tr);
    }
    // Update aria-sort on headers
    document.querySelectorAll('#speakersTable thead th[data-sort-key]').forEach(th => {
      const k = th.getAttribute('data-sort-key');
      th.setAttribute('aria-sort', k === key ? (dir === 'asc' ? 'ascending' : 'descending') : 'none');
    });
  }

  // --------------------- Render: Lista de eventos ---------------------
  function renderEventsList() {
    const tbody = document.getElementById('eventsTableBody');
    tbody.innerHTML = '';
    const events = state.events.slice().sort(sortEventsByDateSlot);
    const filtered = events.filter(matchesFiltersEvent);
    for (const ev of filtered) {
      const spNames = ev.speakers.map(id => getSpeakerById(id)?.nombre || id).join(', ');
      const tr = document.createElement('tr');
      const title = `${ev.tematica || ''} • ${ev.nivel}`;
      tr.innerHTML = `
        <td title="${title}">${formatISO_DDMMYYYY(ev.fecha)}</td>
        <td>${ev.franja}</td>
        <td>${spNames}</td>
        <td>${ev.tematica || ''}</td>
        <td>${ev.nivel}</td>
        <td>${ev.descripcion || ''}</td>
        <td>
          <button class="btn btn-secondary btn-edit" data-id="${ev.id}">Editar</button>
          <button class="btn btn-secondary btn-delete" data-id="${ev.id}">Borrar</button>
        </td>`;
      tbody.appendChild(tr);
    }
    // Wire actions
    tbody.querySelectorAll('.btn-edit').forEach(btn => btn.addEventListener('click', onEditEventClick));
    tbody.querySelectorAll('.btn-delete').forEach(btn => btn.addEventListener('click', onDeleteEventClick));
  }

  // --------------------- Render: Calendario ---------------------
  function renderCalendar() {
    const grid = document.getElementById('calendarGrid');
    grid.innerHTML = '';
    const base = state.startISO || adjustToMonday(getTodayISOInBA());
    state.startISO = adjustToMonday(base);
    const days = buildFourWeekRange(state.startISO);

    for (let i = 0; i < days.length; i++) {
      const iso = days[i];
      const w = weekdayMonday0(iso);
      const dayNum = parseISO(iso).d;
      const dayEl = document.createElement('div');
      dayEl.className = 'calendar-day';
      dayEl.setAttribute('data-iso', iso);

      const header = document.createElement('div');
      header.className = 'day-header';
      header.innerHTML = `<div class="wd">${weekdayShortEs(w)}</div><div class="date">${pad2(dayNum)}/${pad2(parseISO(iso).m)}</div>`;
      dayEl.appendChild(header);

      const franjasEl = document.createElement('div');
      franjasEl.className = 'franjas';

      for (const fr of FRANJAS) {
        const row = document.createElement('div');
        row.className = 'franja-row';
        const label = document.createElement('div');
        label.className = 'franja-label';
        label.textContent = fr;
        row.appendChild(label);

        const evs = state.events.filter(e => e.fecha === iso && e.franja === fr).sort(sortEventsByDateSlot);
        if (evs.length === 0) {
          const disp = document.createElement('div');
          disp.className = 'tile-disponible';
          disp.textContent = 'Disponible';
          row.appendChild(disp);
        } else {
          // determinar clase por nivel o múltiple
          const levels = uniqueLevels(evs);
          const speakersAll = evs.map(e => e.speakers.map(id => getSpeakerById(id)?.nombre || id).join(', ')).join(', ');
          const tile = document.createElement('div');
          tile.className = 'tile-event';
          // estilo por nivel o múltiple
          if (levels.length === 1) {
            const lv = levels[0];
            if (lv === 'Bajo') tile.classList.add('bajo');
            else if (lv === 'Intermedio') tile.classList.add('intermedio');
            else if (lv === 'Alto') tile.classList.add('alto');
          } else {
            // borde con gradiente
            const palette = levels.map(lv => (
              lv === 'Bajo' ? 'var(--bajo-border)' : lv === 'Intermedio' ? 'var(--intermedio-border)' : 'var(--alto-border)'
            ));
            tile.style.backgroundImage = `linear-gradient(#fff,#fff), linear-gradient(90deg, ${palette.join(', ')})`;
            tile.style.backgroundOrigin = 'border-box';
            tile.style.backgroundClip = 'padding-box, border-box';
            tile.style.border = '2px solid transparent';
          }

          // Tooltip: concatenamos info principal
          const tip = evs.map(e => `${e.tematica || ''} • ${e.nivel}`).join(' | ');
          tile.title = tip;

          const anyMatch = evs.some(matchesFiltersEvent);
          if (state.filters.speakerId || state.filters.tematica || state.filters.nivel) {
            if (anyMatch) tile.classList.add('match');
          }

          const ocupado = document.createElement('span');
          ocupado.className = 'ocupado';
          ocupado.textContent = 'Ocupado';
          const speakersSpan = document.createElement('span');
          speakersSpan.className = 'speakers';
          speakersSpan.textContent = speakersAll;
          tile.appendChild(ocupado);
          tile.appendChild(speakersSpan);

          if (evs.length > 1) {
            const count = document.createElement('div');
            count.className = 'count';
            count.textContent = `×${evs.length}`;
            tile.appendChild(count);

            const badges = document.createElement('div');
            badges.className = 'level-badges';
            uniqueLevels(evs).forEach(lv => {
              const b = document.createElement('div');
              b.className = 'level-badge ' + (lv === 'Bajo' ? 'bajo' : lv === 'Intermedio' ? 'intermedio' : 'alto');
              badges.appendChild(b);
            });
            tile.appendChild(badges);
          }

          row.appendChild(tile);
        }

        franjasEl.appendChild(row);
      }

      dayEl.appendChild(franjasEl);
      grid.appendChild(dayEl);
    }
  }

  // --------------------- Modal edición ---------------------
  let lastFocusedBeforeModal = null;
  function openModal() {
    lastFocusedBeforeModal = document.activeElement;
    document.getElementById('modalBackdrop').hidden = false;
    const modal = document.getElementById('eventModal');
    modal.hidden = false;
    // Focus trap
    setTimeout(() => {
      const firstInput = modal.querySelector('input, select, textarea, button');
      firstInput?.focus();
    }, 0);
    document.addEventListener('keydown', trapTabInModal);
  }
  function closeModal() {
    document.getElementById('modalBackdrop').hidden = true;
    const modal = document.getElementById('eventModal');
    modal.hidden = true;
    document.removeEventListener('keydown', trapTabInModal);
    lastFocusedBeforeModal?.focus();
  }
  function trapTabInModal(e) {
    if (e.key !== 'Tab') return;
    const modal = document.getElementById('eventModal');
    const focusables = modal.querySelectorAll('a, button, input, select, textarea, [tabindex]:not([tabindex="-1"])');
    if (focusables.length === 0) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      last.focus();
      e.preventDefault();
    } else if (!e.shiftKey && document.activeElement === last) {
      first.focus();
      e.preventDefault();
    }
  }

  function fillEditModal(ev) {
    document.getElementById('editEvtId').value = ev.id;
    document.getElementById('editEvtFecha').value = ev.fecha;
    document.getElementById('editEvtFranja').value = ev.franja;
    speakersToOptions(document.getElementById('editEvtSpeakers'), ev.speakers);
    document.getElementById('editEvtTematica').value = ev.tematica || '';
    document.getElementById('editEvtDescripcion').value = ev.descripcion || '';
    document.getElementById('editEvtNivel').value = ev.nivel;
    updateCounter('editEvtDescripcion', 'editEvtDescCounter', 200);
  }

  // --------------------- Handlers ---------------------
  function onSpeakerSortClick(e) {
    const th = e.currentTarget;
    const key = th.getAttribute('data-sort-key');
    const { sortSpeakers } = state;
    if (sortSpeakers.key === key) {
      sortSpeakers.dir = sortSpeakers.dir === 'asc' ? 'desc' : 'asc';
    } else {
      sortSpeakers.key = key;
      sortSpeakers.dir = 'asc';
    }
    renderSpeakersTable();
  }

  function onCreateSpeaker(e) {
    e.preventDefault();
    const nombre = document.getElementById('spkNombre').value.trim();
    const tematica = document.getElementById('spkTematica').value.trim();
    const bio = document.getElementById('spkBio').value.trim();
    if (!nombre) {
      showToast('El nombre es obligatorio', 'error');
      return;
    }
    const sp = { id: generateId('spk'), nombre, tematica, bio };
    state.speakers.push(sp);
    saveSpeakersToLS(state.speakers);
    // refresh selects y tabla
    refreshSpeakersDependentUI();
    renderSpeakersTable();
    showToast('Speaker agregado');
    e.target.reset();
  }

  function getMultiSelectValues(select) {
    return Array.from(select.selectedOptions).map(o => o.value);
  }

  function onCreateEvent(e) {
    e.preventDefault();
    const fecha = document.getElementById('evtFecha').value;
    const franja = document.getElementById('evtFranja').value;
    const speakers = getMultiSelectValues(document.getElementById('evtSpeakers'));
    const tematica = document.getElementById('evtTematica').value.trim();
    const descripcion = document.getElementById('evtDescripcion').value.trim();
    const nivel = document.getElementById('evtNivel').value;

    if (!fecha || !FRANJAS.includes(franja) || speakers.length === 0) {
      showToast('Revisa los campos obligatorios', 'error');
      return;
    }

    const conflicts = checkSlotOccupied(fecha, franja);
    if (conflicts.length > 0) {
      const ok = confirm('Ya existe evento en esa fecha/franja. ¿Apilar de todos modos?');
      if (!ok) return;
    }

    const ev = { id: generateId('evt'), fecha, franja, speakers, tematica, descripcion, nivel };
    state.events.push(ev);
    saveEventsToLS(state.events);
    renderEventsList();
    renderCalendar();
    showToast('Evento creado');
    e.target.reset();
    // reset fecha por defecto a hoy
    document.getElementById('evtFecha').value = getTodayISOInBA();
  }

  function onEditEventClick(e) {
    const id = e.currentTarget.getAttribute('data-id');
    const ev = state.events.find(x => x.id === id);
    if (!ev) return;
    fillEditModal(ev);
    openModal();
  }

  function onDeleteEventClick(e) {
    const id = e.currentTarget.getAttribute('data-id');
    const ev = state.events.find(x => x.id === id);
    if (!ev) return;
    const ok = confirm('¿Borrar este evento?');
    if (!ok) return;
    state.events = state.events.filter(x => x.id !== id);
    saveEventsToLS(state.events);
    renderEventsList();
    renderCalendar();
    showToast('Evento borrado');
  }

  function onSaveEditEvent(e) {
    e.preventDefault();
    const id = document.getElementById('editEvtId').value;
    const fecha = document.getElementById('editEvtFecha').value;
    const franja = document.getElementById('editEvtFranja').value;
    const speakers = getMultiSelectValues(document.getElementById('editEvtSpeakers'));
    const tematica = document.getElementById('editEvtTematica').value.trim();
    const descripcion = document.getElementById('editEvtDescripcion').value.trim();
    const nivel = document.getElementById('editEvtNivel').value;

    if (!fecha || !FRANJAS.includes(franja) || speakers.length === 0) {
      showToast('Revisa los campos obligatorios', 'error');
      return;
    }

    const conflicts = checkSlotOccupied(fecha, franja, id);
    if (conflicts.length > 0) {
      const ok = confirm('Ya existe evento en esa fecha/franja. ¿Apilar de todos modos?');
      if (!ok) return;
    }

    const idx = state.events.findIndex(e => e.id === id);
    if (idx >= 0) {
      state.events[idx] = { id, fecha, franja, speakers, tematica, descripcion, nivel };
      saveEventsToLS(state.events);
      renderEventsList();
      renderCalendar();
      showToast('Evento actualizado');
      closeModal();
    }
  }

  function onFiltersChange() {
    state.filters.speakerId = document.getElementById('filterSpeaker').value;
    state.filters.tematica = document.getElementById('filterTematica').value.trim();
    state.filters.nivel = document.getElementById('filterNivel').value;
    renderSpeakersTable();
    renderEventsList();
    renderCalendar();
  }

  function onResetFilters() {
    document.getElementById('filterSpeaker').value = '';
    document.getElementById('filterTematica').value = '';
    document.getElementById('filterNivel').value = '';
    onFiltersChange();
  }

  function onPrevNext(deltaWeeks) {
    state.startISO = addDaysISO(state.startISO, deltaWeeks * 7);
    state.startISO = adjustToMonday(state.startISO);
    document.getElementById('startDate').value = state.startISO;
    renderCalendar();
  }

  function onStartDateChange() {
    const date = document.getElementById('startDate').value;
    state.startISO = adjustToMonday(date || getTodayISOInBA());
    document.getElementById('startDate').value = state.startISO;
    renderCalendar();
  }

  function updateCounter(textareaId, counterId, max) {
    const el = document.getElementById(textareaId);
    const cnt = document.getElementById(counterId);
    const len = (el.value || '').length;
    cnt.textContent = `${len}/${max}`;
  }

  // --------------------- Export/Import ---------------------
  function download(filename, text) {
    const blob = new Blob([text], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function handleExport() {
    const data = { speakers: state.speakers, eventos: state.events };
    download('qff-export.json', JSON.stringify(data, null, 2));
    showToast('Exportado JSON');
  }

  function validateSchema(data) {
    if (!data || typeof data !== 'object') return 'JSON raíz inválido';
    if (!Array.isArray(data.speakers)) return 'Falta array speakers';
    if (!Array.isArray(data.eventos)) return 'Falta array eventos';
    for (const sp of data.speakers) {
      if (!sp.id || !sp.nombre) return 'Speaker inválido: requiere id y nombre';
    }
    for (const ev of data.eventos) {
      if (!ev.id || !ev.fecha || !FRANJAS.includes(ev.franja) || !Array.isArray(ev.speakers)) {
        return 'Evento inválido: requiere id, fecha, franja válida y speakers[]';
      }
      if (ev.nivel && !NIVELES.includes(ev.nivel)) return 'Nivel inválido en evento';
    }
    // Validar que speakers de eventos existan
    const spIds = new Set(data.speakers.map(s => s.id));
    for (const ev of data.eventos) {
      for (const sid of ev.speakers) {
        if (!spIds.has(sid)) return `Evento ${ev.id} referencia speaker inexistente: ${sid}`;
      }
    }
    return null;
  }

  function handleImportFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result);
        const err = validateSchema(data);
        if (err) { showToast(err, 'error'); return; }
        state.speakers = data.speakers;
        state.events = data.eventos;
        saveSpeakersToLS(state.speakers);
        saveEventsToLS(state.events);
        refreshSpeakersDependentUI();
        renderSpeakersTable();
        renderEventsList();
        renderCalendar();
        showToast('Importación exitosa');
      } catch (ex) {
        showToast('JSON inválido', 'error');
      }
    };
    reader.readAsText(file);
  }

  function refreshSpeakersDependentUI() {
    // selects de filtros y formularios
    const filterSpeaker = document.getElementById('filterSpeaker');
    const currVal = filterSpeaker.value;
    filterSpeaker.innerHTML = '<option value="">Todos</option>';
    state.speakers.forEach(sp => {
      const opt = document.createElement('option');
      opt.value = sp.id; opt.textContent = sp.nombre; filterSpeaker.appendChild(opt);
    });
    filterSpeaker.value = currVal;

    speakersToOptions(document.getElementById('evtSpeakers'));
    speakersToOptions(document.getElementById('editEvtSpeakers'));
  }

  // --------------------- Init ---------------------
  function init() {
    seedIfEmpty();
    const data = loadFromLS();
    state.speakers = data.speakers;
    state.events = data.events;

    // Defaults
    const today = getTodayISOInBA();
    state.startISO = adjustToMonday(today);

    // Wire UI
    document.getElementById('evtFecha').value = today;

    document.getElementById('startDate').value = state.startISO;

    document.getElementById('evtDescripcion').addEventListener('input', () => updateCounter('evtDescripcion', 'evtDescCounter', 200));
    document.getElementById('editEvtDescripcion').addEventListener('input', () => updateCounter('editEvtDescripcion', 'editEvtDescCounter', 200));

    document.querySelectorAll('#speakersTable thead th[data-sort-key]').forEach(th => {
      th.addEventListener('click', onSpeakerSortClick);
      th.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSpeakerSortClick({ currentTarget: th }); } });
    });

    document.getElementById('formSpeaker').addEventListener('submit', onCreateSpeaker);
    document.getElementById('formEvent').addEventListener('submit', onCreateEvent);

    document.getElementById('filterSpeaker').addEventListener('change', onFiltersChange);
    document.getElementById('filterTematica').addEventListener('input', onFiltersChange);
    document.getElementById('filterNivel').addEventListener('change', onFiltersChange);
    document.getElementById('resetFiltersBtn').addEventListener('click', onResetFilters);

    document.getElementById('prevBnt');
    document.getElementById('prevBtn').addEventListener('click', () => onPrevNext(-4));
    document.getElementById('nextBtn').addEventListener('click', () => onPrevNext(4));
    document.getElementById('startDate').addEventListener('change', onStartDateChange);

    document.getElementById('importJsonBtn').addEventListener('click', () => document.getElementById('importJsonFile').click());
    document.getElementById('importJsonFile').addEventListener('change', handleImportFile);
    document.getElementById('exportJsonBtn').addEventListener('click', handleExport);

    document.getElementById('modalCloseBtn').addEventListener('click', closeModal);
    document.getElementById('modalCancelBtn').addEventListener('click', closeModal);
    document.getElementById('formEditEvent').addEventListener('submit', onSaveEditEvent);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !document.getElementById('eventModal').hidden) closeModal(); });

    refreshSpeakersDependentUI();
    renderSpeakersTable();
    renderEventsList();
    renderCalendar();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
