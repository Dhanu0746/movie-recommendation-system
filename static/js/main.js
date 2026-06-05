/* ═══════════════════════════════════════
   CineAI – Frontend Logic
═══════════════════════════════════════ */

const API = '';   // same-origin Flask

/* ══════════════════════════════════════════════════════
   STATE
══════════════════════════════════════════════════════ */
let state = {
  users: [],
  items: [],
  stats: {},
  selectedUser: null,
  selectedRating: 0,
  recType: 'hybrid',
  genreFilter: 'all',
  genreChart: null,
  adminGenreChart: null,
  adminTopChart: null,
};

/* ══════════════════════════════════════════════════════
   GENRE EMOJI MAP
══════════════════════════════════════════════════════ */
const GENRE_EMOJI = {
  'Sci-Fi':'🚀','Action':'💥','Crime':'🔫','Drama':'🎭',
  'Romance':'💕','Fantasy':'🧙','Adventure':'🗺️','Thriller':'🔪',
  'Comedy':'😂','Horror':'👻','Animation':'🎨','Default':'🎬',
};
const genreEmoji = g => GENRE_EMOJI[g] || GENRE_EMOJI.Default;

const GENRE_COLOR = {
  'Sci-Fi':'#7c3aed','Action':'#ef4444','Crime':'#6b7280','Drama':'#10b981',
  'Romance':'#ec4899','Fantasy':'#8b5cf6','Adventure':'#f59e0b',
  'Thriller':'#374151','Comedy':'#fbbf24','Horror':'#dc2626',
};
const genreColor = g => GENRE_COLOR[g] || '#6366f1';

/* ══════════════════════════════════════════════════════
   UTILS
══════════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);
const html = (el, h) => { el.innerHTML = h; };

function toast(msg, type = 'success') {
  const t = $('toast');
  t.textContent = msg;
  t.className = `toast ${type}`;
  t.classList.remove('hidden');
  setTimeout(() => t.classList.add('hidden'), 3000);
}

function sentimentBadge(s) {
  if (!s) return '';
  const cls = s.sentiment_label === 'Positive' ? 'sentiment-pos'
             : s.sentiment_label === 'Negative' ? 'sentiment-neg' : 'sentiment-neu';
  return `<span class="card-sentiment ${cls}">${s.sentiment_label === 'Positive' ? '😊' : s.sentiment_label === 'Negative' ? '😞' : '😐'} ${s.sentiment_label} (${s.review_count} reviews)</span>`;
}

function stars(rating) {
  return '★'.repeat(Math.round(rating)) + '☆'.repeat(5 - Math.round(rating));
}

/* ══════════════════════════════════════════════════════
   API CALLS
══════════════════════════════════════════════════════ */
async function fetchJSON(url) {
  const res = await fetch(API + url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function postJSON(url, body) {
  const res = await fetch(API + url, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/* ══════════════════════════════════════════════════════
   CARD BUILDER
══════════════════════════════════════════════════════ */
function buildCard(movie, opts = {}) {
  const { score, reason, relevance } = opts;
  const avgR = movie.average_rating || 0;
  const sem  = movie.sentiment;
  const scoreStr = score != null
    ? `<div class="card-score">🤖 AI Score: ${(score * 100).toFixed(1)}%</div>`
    : relevance != null
    ? `<div class="card-score">🔍 Match: ${(relevance * 100).toFixed(1)}%</div>`
    : avgR > 0
    ? `<div class="card-score">★ ${avgR.toFixed(1)}</div>`
    : '';
  const reasonStr = reason
    ? `<div class="card-reason">💡 ${reason}</div>` : '';

  const card = document.createElement('div');
  card.className = 'movie-card';
  card.dataset.id = movie.item_id;
  card.innerHTML = `
    <div class="card-poster">${genreEmoji(movie.genre)}</div>
    <div class="card-body">
      <div class="card-title">${movie.title}</div>
      <div class="card-meta">
        <span class="card-genre">${movie.genre}</span>
        <span>${movie.year || ''}</span>
      </div>
      ${scoreStr}
      ${sentimentBadge(sem)}
      ${reasonStr}
    </div>`;
  card.addEventListener('click', () => openMovieModal(movie.item_id));
  return card;
}

/* ══════════════════════════════════════════════════════
   MOVIE MODAL
══════════════════════════════════════════════════════ */
async function openMovieModal(itemId) {
  try {
    const movie = await fetchJSON(`/api/items/${itemId}`);
    const sem   = movie.sentiment || {};
    const posW  = sem.positive_pct || 50;
    const negW  = sem.negative_pct || 50;
    const avgR  = movie.average_rating || 0;

    const similarHTML = (movie.similar || []).map(s =>
      `<span class="similar-chip" data-id="${s.item_id}">${genreEmoji(s.genre)} ${s.title}</span>`
    ).join('');

    $('modalBody').innerHTML = `
      <div class="modal-header">
        <div class="modal-poster">${genreEmoji(movie.genre)}</div>
        <div>
          <div class="modal-title">${movie.title}</div>
          <div class="modal-meta">
            🎬 ${movie.genre} &nbsp;•&nbsp; 📅 ${movie.year || '—'}<br/>
            🎥 ${movie.director || '—'}<br/>
            ★ ${avgR > 0 ? avgR.toFixed(1) : 'No ratings yet'}
          </div>
        </div>
      </div>

      <div class="modal-section">
        <h4>Overview</h4>
        <p class="modal-desc">${movie.description || 'No description available.'}</p>
      </div>

      <div class="modal-section">
        <h4>Sentiment Analysis</h4>
        <div style="display:flex;gap:1.5rem;align-items:center;font-size:.88rem;">
          <span class="${sem.sentiment_label === 'Positive' ? 'sentiment-pos' : sem.sentiment_label === 'Negative' ? 'sentiment-neg' : ''}" style="font-weight:700">
            ${sem.sentiment_label === 'Positive' ? '😊' : sem.sentiment_label === 'Negative' ? '😞' : '😐'}
            ${sem.sentiment_label || 'Neutral'}
          </span>
          <span style="color:var(--text2)">Score: ${sem.sentiment_score != null ? sem.sentiment_score.toFixed(2) : '—'}</span>
          <span style="color:var(--text2)">${sem.review_count || 0} reviews</span>
        </div>
        <div class="sentiment-bar-wrap">
          <span style="font-size:.75rem;color:var(--text3)">Positive ${posW}%</span>
          <div class="sentiment-bar-bg"><div class="sentiment-bar pos" style="width:${posW}%"></div></div>
        </div>
        <div class="sentiment-bar-wrap" style="margin-top:.35rem">
          <span style="font-size:.75rem;color:var(--text3)">Negative ${negW}%</span>
          <div class="sentiment-bar-bg"><div class="sentiment-bar neg" style="width:${negW}%"></div></div>
        </div>
      </div>

      ${movie.similar && movie.similar.length ? `
      <div class="modal-section">
        <h4>Similar Movies</h4>
        <div class="similar-grid">${similarHTML}</div>
      </div>` : ''}`;

    $('movieModal').classList.remove('hidden');

    // Wire similar chips
    $('movieModal').querySelectorAll('.similar-chip').forEach(chip => {
      chip.addEventListener('click', (e) => {
        e.stopPropagation();
        $('movieModal').classList.add('hidden');
        openMovieModal(parseInt(chip.dataset.id));
      });
    });
  } catch (err) {
    toast('Failed to load movie details', 'error');
  }
}

$('modalClose').addEventListener('click', () => $('movieModal').classList.add('hidden'));
$('movieModal').addEventListener('click', e => {
  if (e.target === $('movieModal')) $('movieModal').classList.add('hidden');
});

/* ══════════════════════════════════════════════════════
   TRENDING CAROUSEL
══════════════════════════════════════════════════════ */
async function loadTrending() {
  try {
    const data = await fetchJSON('/api/trending?n=12');
    const carousel = $('trendingCarousel');
    carousel.innerHTML = '';
    (data.trending || []).forEach(movie => {
      carousel.appendChild(buildCard(movie, { score: movie.popularity_score / 5 }));
    });
  } catch (e) { console.error('Trending error:', e); }
}

$('trendPrev').addEventListener('click', () => {
  $('trendingCarousel').scrollBy({ left: -650, behavior: 'smooth' });
});
$('trendNext').addEventListener('click', () => {
  $('trendingCarousel').scrollBy({ left: 650, behavior: 'smooth' });
});

/* ══════════════════════════════════════════════════════
   GENRE PILLS (for trending filter)
══════════════════════════════════════════════════════ */
function buildGenrePills(genres) {
  const pills = $('genrePills');
  pills.innerHTML = '';
  genres.forEach(g => {
    const p = document.createElement('button');
    p.className = 'genre-pill';
    p.textContent = `${genreEmoji(g)} ${g}`;
    p.addEventListener('click', async () => {
      document.querySelectorAll('.genre-pill').forEach(x => x.classList.remove('active'));
      p.classList.add('active');
      if (g === 'All') { loadTrending(); return; }
      const data = await fetchJSON(`/api/trending/genre?genre=${encodeURIComponent(g)}&n=10`);
      const carousel = $('trendingCarousel');
      carousel.innerHTML = '';
      (data.trending || []).forEach(movie => carousel.appendChild(buildCard(movie, {})));
    });
    pills.appendChild(p);
  });
}

/* ══════════════════════════════════════════════════════
   ALL MOVIES GRID
══════════════════════════════════════════════════════ */
async function loadAllMovies() {
  try {
    const movies = await fetchJSON('/api/items');
    state.items = movies;
    renderAllMovies(movies);

    // Build genre filter pills
    const genres = ['all', ...new Set(movies.map(m => m.genre))];
    const row = $('genreFilterRow');
    row.innerHTML = '';
    genres.forEach(g => {
      const p = document.createElement('button');
      p.className = 'filter-pill' + (g === 'all' ? ' active' : '');
      p.textContent = g === 'all' ? '🎬 All' : `${genreEmoji(g)} ${g}`;
      p.addEventListener('click', () => {
        document.querySelectorAll('.filter-pill').forEach(x => x.classList.remove('active'));
        p.classList.add('active');
        state.genreFilter = g;
        renderAllMovies(g === 'all' ? movies : movies.filter(m => m.genre === g));
      });
      row.appendChild(p);
    });

    return movies;
  } catch (e) { console.error('Items error:', e); }
}

function renderAllMovies(movies) {
  const grid = $('allMoviesGrid');
  grid.innerHTML = '';
  movies.forEach(m => grid.appendChild(buildCard(m)));
}

/* ══════════════════════════════════════════════════════
   USERS
══════════════════════════════════════════════════════ */
async function loadUsers() {
  try {
    const users = await fetchJSON('/api/users');
    state.users = users;

    // Populate both user selects
    const navSel = $('globalUserSelect');
    navSel.innerHTML = '<option value="">Select User</option>';
    const rateSel = $('rateMovieSelect');

    users.forEach(u => {
      const opt = document.createElement('option');
      opt.value = u.user_id;
      opt.textContent = u.name;
      navSel.appendChild(opt);
    });

    navSel.addEventListener('change', () => {
      const uid = parseInt(navSel.value);
      if (!uid) return;
      state.selectedUser = users.find(u => u.user_id === uid);
      onUserSelected();
    });

    // Populate rate-movie select
    populateRateSelect();
  } catch (e) { console.error('Users error:', e); }
}

function populateRateSelect() {
  const sel = $('rateMovieSelect');
  sel.innerHTML = '<option value="">Choose a movie…</option>';
  state.items.forEach(m => {
    const o = document.createElement('option');
    o.value = m.item_id;
    o.textContent = m.title;
    sel.appendChild(o);
  });
}

function onUserSelected() {
  const user = state.selectedUser;
  if (!user) return;

  // Update profile card
  $('profileAvatar').textContent = user.name[0].toUpperCase();
  $('profileName').textContent   = user.name;
  $('profileEmail').textContent  = user.email || '';

  const ratingCount = user.rating_count || Object.keys(user.ratings || {}).length || 0;
  const avgRating   = user.average_rating || 0;
  $('profileRatingCount').textContent = ratingCount;
  $('profileAvgRating').textContent   = avgRating ? avgRating.toFixed(1) : '—';

  // Watch history
  buildWatchHistory(user);
  // Genre chart
  buildGenreChart(user);
  // Load recommendations
  loadRecommendations();
}

function buildWatchHistory(user) {
  const list  = $('watchHistoryList');
  const rated = user.ratings || {};
  list.innerHTML = '';
  const entries = Object.entries(rated).sort((a,b) => b[1] - a[1]);
  if (entries.length === 0) { list.innerHTML = '<div style="color:var(--text3);font-size:.85rem">No ratings yet</div>'; return; }
  entries.slice(0, 8).forEach(([iid, r]) => {
    const movie = state.items.find(m => m.item_id === parseInt(iid));
    if (!movie) return;
    const div = document.createElement('div');
    div.className = 'history-item';
    div.innerHTML = `<span>${genreEmoji(movie.genre)} ${movie.title}</span><span class="history-stars">${stars(r)}</span>`;
    div.addEventListener('click', () => openMovieModal(parseInt(iid)));
    list.appendChild(div);
  });
}

function buildGenreChart(user) {
  const rated = user.ratings || {};
  const genreCounts = {};
  Object.keys(rated).forEach(iid => {
    const movie = state.items.find(m => m.item_id === parseInt(iid));
    if (movie) genreCounts[movie.genre] = (genreCounts[movie.genre] || 0) + 1;
  });

  const labels = Object.keys(genreCounts);
  const data   = Object.values(genreCounts);
  const colors = labels.map(genreColor);

  if (state.genreChart) state.genreChart.destroy();
  const ctx = document.getElementById('genreChart').getContext('2d');
  state.genreChart = new Chart(ctx, {
    type: 'doughnut',
    data: { labels, datasets: [{ data, backgroundColor: colors, borderWidth: 0 }] },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 11 } } },
      },
    },
  });
}

/* ══════════════════════════════════════════════════════
   RECOMMENDATIONS
══════════════════════════════════════════════════════ */
async function loadRecommendations() {
  const user = state.selectedUser;
  if (!user) return;

  $('aiEmptyState').classList.add('hidden');
  $('recsGrid').classList.remove('hidden');
  $('recsGrid').innerHTML = '<div style="color:var(--text2);padding:2rem">Loading AI recommendations…</div>';

  try {
    const data = await fetchJSON(`/api/recommendations/${user.user_id}?type=${state.recType}&n=9`);
    const grid = $('recsGrid');
    grid.innerHTML = '';
    if (data.cold_start) {
      grid.insertAdjacentHTML('beforeend', '<div style="color:var(--accent2);font-size:.85rem;margin-bottom:1rem">👋 Welcome! Showing popular movies until you rate some films.</div>');
    }
    (data.recommendations || []).forEach(rec => {
      const card = buildCard(rec.item, { score: rec.score, reason: rec.reason });
      grid.appendChild(card);
    });
    if (!data.recommendations?.length) {
      grid.innerHTML = '<div style="color:var(--text2);padding:2rem">No recommendations available yet. Try rating some movies!</div>';
    }
  } catch (e) {
    $('recsGrid').innerHTML = '<div style="color:var(--red);padding:2rem">Failed to load recommendations.</div>';
  }
}

// Rec type switcher
document.querySelectorAll('.rec-type-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.rec-type-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.recType = btn.dataset.type;
    loadRecommendations();
  });
});

/* ══════════════════════════════════════════════════════
   SEARCH
══════════════════════════════════════════════════════ */
async function doSearch(q) {
  if (!q.trim()) return;
  try {
    const data = await fetchJSON(`/api/search?q=${encodeURIComponent(q)}&n=12`);
    const sec  = $('searchResults');
    const grid = $('searchResultsGrid');
    $('searchResultsTag').textContent = `${data.results.length} results for "${q}"`;
    grid.innerHTML = '';
    (data.results || []).forEach(movie => grid.appendChild(buildCard(movie, { relevance: movie.relevance })));
    sec.classList.remove('hidden');
    sec.scrollIntoView({ behavior: 'smooth' });
  } catch (e) { toast('Search failed', 'error'); }
}

$('heroSearchBtn').addEventListener('click', () => doSearch($('heroSearchInput').value));
$('heroSearchInput').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(e.target.value); });

/* ══════════════════════════════════════════════════════
   STAR RATING
══════════════════════════════════════════════════════ */
document.querySelectorAll('.star').forEach(star => {
  star.addEventListener('click', () => {
    state.selectedRating = parseInt(star.dataset.val);
    document.querySelectorAll('.star').forEach((s, i) => {
      s.classList.toggle('active', i < state.selectedRating);
    });
  });
  star.addEventListener('mouseover', () => {
    const v = parseInt(star.dataset.val);
    document.querySelectorAll('.star').forEach((s, i) => s.classList.toggle('active', i < v));
  });
});
$('starRow').addEventListener('mouseleave', () => {
  document.querySelectorAll('.star').forEach((s, i) => {
    s.classList.toggle('active', i < state.selectedRating);
  });
});

$('submitRatingBtn').addEventListener('click', async () => {
  const user   = state.selectedUser;
  const itemId = parseInt($('rateMovieSelect').value);
  const rating = state.selectedRating;
  const msg    = $('ratingMsg');

  if (!user)   { msg.textContent = 'Select a user first!'; msg.className = 'rating-msg error'; return; }
  if (!itemId) { msg.textContent = 'Select a movie!';      msg.className = 'rating-msg error'; return; }
  if (!rating) { msg.textContent = 'Select a star rating!';msg.className = 'rating-msg error'; return; }

  try {
    await postJSON('/api/ratings', { user_id: user.user_id, item_id: itemId, rating });
    msg.textContent = '✅ Rating submitted! Recommendations updated.';
    msg.className   = 'rating-msg success';
    toast('Rating saved! AI is updating your recommendations…');
    // Refresh user data
    const updated = await fetchJSON(`/api/users/${user.user_id}`);
    state.selectedUser = updated;
    onUserSelected();
    state.selectedRating = 0;
    document.querySelectorAll('.star').forEach(s => s.classList.remove('active'));
  } catch (e) {
    msg.textContent = 'Failed to submit rating.';
    msg.className   = 'rating-msg error';
  }
});

/* ══════════════════════════════════════════════════════
   STATS & ADMIN DASHBOARD
══════════════════════════════════════════════════════ */
async function loadStats() {
  try {
    const s = await fetchJSON('/api/stats');
    state.stats = s;

    // Hero stats
    document.querySelector('#statUsers .stat-num').textContent  = s.total_users;
    document.querySelector('#statMovies .stat-num').textContent = s.total_items;
    document.querySelector('#statRatings .stat-num').textContent= s.total_ratings;

    // Admin metrics strip
    $('mUsers').textContent  = s.total_users;
    $('mMovies').textContent = s.total_items;
    $('mRatings').textContent= s.total_ratings;
    $('mAvg').textContent    = s.average_rating;

    // Admin genre chart
    buildAdminGenreChart(s.genre_distribution);
    // Admin top movies chart
    buildAdminTopMoviesChart(s.most_rated_movies);
    // Admin top movies grid
    renderAdminMoviesGrid(s.most_rated_movies);
  } catch (e) { console.error('Stats error:', e); }
}

function buildAdminGenreChart(genreDist) {
  const labels = Object.keys(genreDist);
  const data   = Object.values(genreDist);
  const colors = labels.map(genreColor);
  if (state.adminGenreChart) state.adminGenreChart.destroy();
  const ctx = document.getElementById('adminGenreChart').getContext('2d');
  state.adminGenreChart = new Chart(ctx, {
    type: 'pie',
    data: { labels, datasets: [{ data, backgroundColor: colors, borderWidth: 0 }] },
    options: { responsive: true, plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } } },
  });
}

function buildAdminTopMoviesChart(movies) {
  if (!movies?.length) return;
  if (state.adminTopChart) state.adminTopChart.destroy();
  const ctx = document.getElementById('adminTopMoviesChart').getContext('2d');
  state.adminTopChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: movies.map(m => m.title.length > 14 ? m.title.slice(0,14)+'…' : m.title),
      datasets: [{
        label: 'Ratings', data: movies.map(m => m.rating_count),
        backgroundColor: movies.map(m => genreColor(m.genre)), borderRadius: 6,
      }],
    },
    options: {
      responsive: true, indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: '#1e1e2e' }, ticks: { color: '#94a3b8' } },
        y: { grid: { display: false },   ticks: { color: '#94a3b8' } },
      },
    },
  });
}

function renderAdminMoviesGrid(movies) {
  const grid = $('adminTopMoviesGrid');
  grid.innerHTML = '';
  (movies || []).forEach(m => grid.appendChild(buildCard(m)));
}

$('runMetricsBtn').addEventListener('click', async () => {
  $('runMetricsBtn').textContent = 'Evaluating…';
  try {
    const m = await fetchJSON('/api/metrics');
    $('mRmse').textContent = m.rmse ?? '—';
    $('mPrec').textContent = m.precision_at_k ?? '—';
    $('mRec').textContent  = m.recall_at_k ?? '—';
    $('mF1').textContent   = m.f1_at_k ?? '—';
    toast(`✅ Evaluation complete on ${m.test_samples} samples`);
  } catch (e) { toast('Evaluation failed', 'error'); }
  finally { $('runMetricsBtn').textContent = 'Run Evaluation'; }
});

/* ══════════════════════════════════════════════════════
   THEME TOGGLE
══════════════════════════════════════════════════════ */
$('themeToggle').addEventListener('click', () => {
  document.body.classList.toggle('light');
  $('themeToggle').textContent = document.body.classList.contains('light') ? '🌞' : '🌙';
});

/* ══════════════════════════════════════════════════════
   NAV ACTIVE LINK
══════════════════════════════════════════════════════ */
document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', e => {
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    link.classList.add('active');
    const section = link.dataset.section;
    // Show/hide dashboard and admin sections
    $('dashboard').classList.toggle('hidden', section !== 'dashboard');
    $('admin').classList.toggle('hidden', section !== 'admin');
    if (section === 'admin') loadStats();
  });
});

/* ══════════════════════════════════════════════════════
   NAVBAR SCROLL EFFECT
══════════════════════════════════════════════════════ */
window.addEventListener('scroll', () => {
  $('navbar').style.background = window.scrollY > 40
    ? 'rgba(10,10,15,0.98)'
    : 'rgba(10,10,15,0.85)';
});

/* ══════════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════════ */
async function init() {
  await loadAllMovies();
  await loadUsers();
  await loadTrending();
  await loadStats();

  // Build genre pills for trending
  const genres = ['All', ...new Set(state.items.map(m => m.genre))];
  buildGenrePills(genres);
  populateRateSelect();
}

init();
