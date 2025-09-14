// Global variables
let currentUser = null;
let users = [];
let items = [];
let currentRecommendationType = 'hybrid';

// DOM elements
const userSelect = document.getElementById('userSelect');
const loadUserBtn = document.getElementById('loadUserBtn');
const userProfile = document.getElementById('userProfile');
const userInfo = document.getElementById('userInfo');
const userRatings = document.getElementById('userRatings');
const recTypeSection = document.getElementById('recTypeSection');
const recommendations = document.getElementById('recommendations');
const recLoading = document.getElementById('recLoading');
const recResults = document.getElementById('recResults');
const similarItems = document.getElementById('similarItems');
const similarLoading = document.getElementById('similarLoading');
const similarResults = document.getElementById('similarResults');
const addRatingSection = document.getElementById('addRatingSection');
const itemSelect = document.getElementById('itemSelect');
const addRatingBtn = document.getElementById('addRatingBtn');
const systemStats = document.getElementById('systemStats');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadUsers();
    loadItems();
    loadSystemStats();
    setupEventListeners();
});

// Event listeners
function setupEventListeners() {
    loadUserBtn.addEventListener('click', loadUserProfile);
    addRatingBtn.addEventListener('click', addRating);
    
    // Training button
    const trainBtn = document.getElementById('trainRealDataBtn');
    if (trainBtn) {
        trainBtn.addEventListener('click', trainOnRealData);
    }
    
    // Recommendation type buttons
    document.querySelectorAll('.rec-type-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.rec-type-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            currentRecommendationType = this.dataset.type;
            if (currentUser) {
                loadRecommendations();
            }
        });
    });
}

// Load users from API
async function loadUsers() {
    try {
        const response = await fetch('/api/users');
        users = await response.json();
        
        userSelect.innerHTML = '<option value="">Choose a user...</option>';
        users.forEach(user => {
            const option = document.createElement('option');
            option.value = user.user_id;
            option.textContent = `${user.name} (ID: ${user.user_id})`;
            userSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading users:', error);
        showError('Failed to load users');
    }
}

// Load items from API
async function loadItems() {
    try {
        const response = await fetch('/api/items');
        items = await response.json();
        
        itemSelect.innerHTML = '<option value="">Select a movie...</option>';
        items.forEach(item => {
            const option = document.createElement('option');
            option.value = item.item_id;
            option.textContent = item.title;
            itemSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading items:', error);
        showError('Failed to load items');
    }
}

// Load user profile
async function loadUserProfile() {
    const userId = userSelect.value;
    if (!userId) {
        showError('Please select a user');
        return;
    }
    
    try {
        currentUser = users.find(u => u.user_id == userId);
        if (!currentUser) {
            showError('User not found');
            return;
        }
        
        displayUserProfile();
        userProfile.classList.remove('hidden');
        recTypeSection.classList.remove('hidden');
        addRatingSection.classList.remove('hidden');
        
        // Load recommendations
        loadRecommendations();
        
    } catch (error) {
        console.error('Error loading user profile:', error);
        showError('Failed to load user profile');
    }
}

// Display user profile
function displayUserProfile() {
    // User info
    userInfo.innerHTML = `
        <h3>${currentUser.name}</h3>
        <p><strong>User ID:</strong> ${currentUser.user_id}</p>
        <p><strong>Email:</strong> ${currentUser.email || 'N/A'}</p>
        <p><strong>Average Rating:</strong> ${currentUser.average_rating.toFixed(2)}/5</p>
        <p><strong>Total Ratings:</strong> ${Object.keys(currentUser.ratings).length}</p>
    `;
    
    // User ratings
    userRatings.innerHTML = '';
    if (Object.keys(currentUser.ratings).length === 0) {
        userRatings.innerHTML = '<p>No ratings yet.</p>';
        return;
    }
    
    Object.entries(currentUser.ratings).forEach(([itemId, rating]) => {
        const item = items.find(i => i.item_id == itemId);
        if (item) {
            const ratingItem = document.createElement('div');
            ratingItem.className = 'rating-item';
            ratingItem.innerHTML = `
                <h4>${item.title}</h4>
                <div class="rating-stars">${generateStars(rating)}</div>
                <div class="rating-value">Rating: ${rating}/5</div>
            `;
            userRatings.appendChild(ratingItem);
        }
    });
}

// Load recommendations
async function loadRecommendations() {
    if (!currentUser) return;
    
    recLoading.classList.remove('hidden');
    recResults.innerHTML = '';
    recommendations.classList.remove('hidden');
    
    try {
        const response = await fetch(`/api/recommendations/${currentUser.user_id}?type=${currentRecommendationType}&n=6`);
        const data = await response.json();
        
        recLoading.classList.add('hidden');
        displayRecommendations(data.recommendations);
        
    } catch (error) {
        console.error('Error loading recommendations:', error);
        recLoading.classList.add('hidden');
        showError('Failed to load recommendations');
    }
}

// Display recommendations
function displayRecommendations(recs) {
    recResults.innerHTML = '';
    
    if (recs.length === 0) {
        recResults.innerHTML = '<p>No recommendations available.</p>';
        return;
    }
    
    recs.forEach(rec => {
        const recCard = document.createElement('div');
        recCard.className = 'rec-card';
        recCard.innerHTML = `
            <h3>${rec.item.title}</h3>
            <div class="rec-meta">
                <strong>Genre:</strong> ${rec.item.genre || 'N/A'} | 
                <strong>Year:</strong> ${rec.item.year || 'N/A'} | 
                <strong>Director:</strong> ${rec.item.director || 'N/A'}
            </div>
            <div class="rec-score">Score: ${rec.score.toFixed(3)}</div>
            <div class="rec-description">${rec.item.description || 'No description available.'}</div>
        `;
        
        // Add click event to show similar items
        recCard.addEventListener('click', () => {
            loadSimilarItems(rec.item.item_id);
        });
        
        recResults.appendChild(recCard);
    });
}

// Load similar items
async function loadSimilarItems(itemId) {
    similarLoading.classList.remove('hidden');
    similarResults.innerHTML = '';
    similarItems.classList.remove('hidden');
    
    try {
        const response = await fetch(`/api/similar/${itemId}?n=6`);
        const data = await response.json();
        
        similarLoading.classList.add('hidden');
        displaySimilarItems(data.similar_items);
        
    } catch (error) {
        console.error('Error loading similar items:', error);
        similarLoading.classList.add('hidden');
        showError('Failed to load similar items');
    }
}

// Display similar items
function displaySimilarItems(similarItems) {
    similarResults.innerHTML = '';
    
    if (similarItems.length === 0) {
        similarResults.innerHTML = '<p>No similar items found.</p>';
        return;
    }
    
    similarItems.forEach(item => {
        const itemCard = document.createElement('div');
        itemCard.className = 'rec-card';
        itemCard.innerHTML = `
            <h3>${item.item.title}</h3>
            <div class="rec-meta">
                <strong>Genre:</strong> ${item.item.genre || 'N/A'} | 
                <strong>Year:</strong> ${item.item.year || 'N/A'} | 
                <strong>Director:</strong> ${item.item.director || 'N/A'}
            </div>
            <div class="rec-score">Similarity: ${item.similarity.toFixed(3)}</div>
            <div class="rec-description">${item.item.description || 'No description available.'}</div>
        `;
        similarResults.appendChild(itemCard);
    });
}

// Add rating
async function addRating() {
    const userId = currentUser?.user_id;
    const itemId = itemSelect.value;
    const rating = document.querySelector('input[name="rating"]:checked')?.value;
    
    if (!userId) {
        showError('Please select a user first');
        return;
    }
    
    if (!itemId) {
        showError('Please select a movie');
        return;
    }
    
    if (!rating) {
        showError('Please select a rating');
        return;
    }
    
    try {
        const response = await fetch('/api/ratings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: parseInt(userId),
                item_id: parseInt(itemId),
                rating: parseInt(rating)
            })
        });
        
        if (response.ok) {
            showSuccess('Rating added successfully!');
            
            // Reset form
            itemSelect.value = '';
            document.querySelectorAll('input[name="rating"]').forEach(input => {
                input.checked = false;
            });
            
            // Reload user profile and recommendations
            loadUserProfile();
            loadSystemStats();
            
        } else {
            const error = await response.json();
            showError(error.error || 'Failed to add rating');
        }
        
    } catch (error) {
        console.error('Error adding rating:', error);
        showError('Failed to add rating');
    }
}

// Load system statistics
async function loadSystemStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        
        systemStats.innerHTML = `
            <div class="stat-card">
                <h3>${stats.total_users}</h3>
                <p>Total Users</p>
            </div>
            <div class="stat-card">
                <h3>${stats.total_items}</h3>
                <p>Total Movies</p>
            </div>
            <div class="stat-card">
                <h3>${stats.total_ratings}</h3>
                <p>Total Ratings</p>
            </div>
            <div class="stat-card">
                <h3>${stats.average_rating.toFixed(2)}</h3>
                <p>Average Rating</p>
            </div>
        `;
        
    } catch (error) {
        console.error('Error loading system stats:', error);
    }
}

// Utility functions
function generateStars(rating) {
    let stars = '';
    for (let i = 1; i <= 5; i++) {
        if (i <= rating) {
            stars += '<i class="fas fa-star"></i>';
        } else {
            stars += '<i class="far fa-star"></i>';
        }
    }
    return stars;
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error';
    errorDiv.textContent = message;
    
    // Remove existing error messages
    document.querySelectorAll('.error').forEach(el => el.remove());
    
    // Add new error message
    document.querySelector('.main-content').insertBefore(errorDiv, document.querySelector('.main-content').firstChild);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

function showSuccess(message) {
    const successDiv = document.createElement('div');
    successDiv.className = 'success';
    successDiv.textContent = message;
    
    // Remove existing success messages
    document.querySelectorAll('.success').forEach(el => el.remove());
    
    // Add new success message
    document.querySelector('.main-content').insertBefore(successDiv, document.querySelector('.main-content').firstChild);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        successDiv.remove();
    }, 3000);
}

// Train on real data
async function trainOnRealData() {
    const trainBtn = document.getElementById('trainRealDataBtn');
    const trainingStatus = document.getElementById('trainingStatus');
    const trainingProgress = document.getElementById('trainingProgress');
    const trainingResults = document.getElementById('trainingResults');
    
    // Show training progress
    trainBtn.disabled = true;
    trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
    trainingStatus.classList.remove('hidden');
    trainingProgress.classList.remove('hidden');
    trainingResults.classList.add('hidden');
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Show success
            trainingProgress.classList.add('hidden');
            trainingResults.classList.remove('hidden');
            trainingResults.innerHTML = `
                <div class="success">
                    <h3>✅ Training Completed Successfully!</h3>
                    <p><strong>Dataset:</strong> ${result.stats.dataset_size.users} users, ${result.stats.dataset_size.items} movies, ${result.stats.dataset_size.ratings} ratings</p>
                    <p><strong>Training Time:</strong> ${result.stats.training_time.collaborative.toFixed(2)}s (Collaborative), ${result.stats.training_time.content.toFixed(2)}s (Content-based)</p>
                    <p>The system is now trained on real MovieLens data!</p>
                </div>
            `;
            
            // Reload data
            loadUsers();
            loadItems();
            loadSystemStats();
            
            if (currentUser) {
                loadRecommendations();
            }
            
        } else {
            throw new Error(result.message || 'Training failed');
        }
        
    } catch (error) {
        console.error('Error training on real data:', error);
        trainingProgress.classList.add('hidden');
        trainingResults.classList.remove('hidden');
        trainingResults.innerHTML = `
            <div class="error">
                <h3>❌ Training Failed</h3>
                <p>Error: ${error.message}</p>
                <p>Please try again or check your internet connection.</p>
            </div>
        `;
    } finally {
        // Reset button
        trainBtn.disabled = false;
        trainBtn.innerHTML = '<i class="fas fa-download"></i> Train on Real MovieLens Data';
    }
}
