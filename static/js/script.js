// Main JavaScript for Rock Paper Scissors Lizard Spock Game

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const captureBtn = document.getElementById('captureBtn');
    const userGesture = document.getElementById('userGesture');
    const computerGesture = document.getElementById('computerGesture');
    const gameResult = document.getElementById('gameResult');
    const userGestureIcon = document.getElementById('userGestureIcon');
    const computerGestureIcon = document.getElementById('computerGestureIcon');
    const classicModeBtn = document.getElementById('classicModeBtn');
    const extendedModeBtn = document.getElementById('extendedModeBtn');
    const classicRules = document.getElementById('classicRules');
    const extendedRules = document.getElementById('extendedRules');
    const lizardGuide = document.getElementById('lizardGuide');
    const spockGuide = document.getElementById('spockGuide');
    
    // Game state
    let gameMode = 'extended'; // 'classic' or 'extended'
    
    // Function to capitalize first letter
    function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    
    // Gesture icons mapping
    const gestureIcons = {
        'rock': '<i class="fas fa-hand-rock"></i>',
        'paper': '<i class="fas fa-hand-paper"></i>',
        'scissors': '<i class="fas fa-hand-scissors"></i>',
        'lizard': '<i class="fas fa-hand-lizard"></i>',
        'spock': '<i class="fas fa-hand-spock"></i>',
        'unknown': '<i class="fas fa-question-circle"></i>'
    };
    
    // Function to get gesture icon
    function getGestureIcon(gesture) {
        return gestureIcons[gesture] || gestureIcons.unknown;
    }
    
    // Function to refresh video feeds to prevent caching
    function refreshVideoFeeds() {
        // Refresh both video feeds by adding a timestamp to force reload
        const timestamp = new Date().getTime();
        
        // Get the video feed elements
        const videoFeed = document.getElementById('video_feed');
        const processedFeed = document.getElementById('processed_feed');
        
        // Update the src attributes with a new timestamp
        if (videoFeed) {
            const currentSrc = videoFeed.src.split('?')[0]; // Remove any existing query params
            videoFeed.src = `${currentSrc}?t=${timestamp}`;
        }
        
        if (processedFeed) {
            const currentSrc = processedFeed.src.split('?')[0]; // Remove any existing query params
            processedFeed.src = `${currentSrc}?t=${timestamp}`;
        }
    }
    
    // Function to update the UI with game results
    function updateGameUI(result) {
        // Update user gesture
        const userGestureText = result.user_gesture === 'unknown' 
            ? 'Not detected' 
            : capitalizeFirstLetter(result.user_gesture);
        userGesture.textContent = userGestureText;
        
        // Update computer gesture
        computerGesture.textContent = capitalizeFirstLetter(result.computer_gesture);
        
        // Update gesture icons
        userGestureIcon.innerHTML = getGestureIcon(result.user_gesture);
        computerGestureIcon.innerHTML = getGestureIcon(result.computer_gesture);
        
        // Refresh video feeds to ensure we're showing the latest frames
        refreshVideoFeeds();
        
        // Add animation class
        userGestureIcon.classList.add('animate-pulse');
        computerGestureIcon.classList.add('animate-pulse');
        
        // Remove animation after a delay
        setTimeout(() => {
            userGestureIcon.classList.remove('animate-pulse');
            computerGestureIcon.classList.remove('animate-pulse');
        }, 1000);
        
        // Update game result
        gameResult.textContent = result.result;
        
        // Change background color based on result
        if (result.result.includes('You win')) {
            gameResult.parentElement.style.backgroundColor = 'var(--success-color)'; // Green for win
        } else if (result.result.includes('Computer wins')) {
            gameResult.parentElement.style.backgroundColor = 'var(--secondary-color)'; // Red for loss
        } else if (result.result.includes('Tie')) {
            gameResult.parentElement.style.backgroundColor = 'var(--warning-color)'; // Orange for tie
        } else {
            // Error or other cases
            gameResult.parentElement.style.backgroundColor = '#7f8c8d'; // Gray for error
        }
    }
    
    // We're only using extended mode now
    gameMode = 'extended';
    
    // Hide classic mode UI elements
    if (classicModeBtn && extendedModeBtn) {
        classicModeBtn.style.display = 'none';
        extendedModeBtn.style.display = 'none';
    }
    
    // Show only extended rules
    if (classicRules && extendedRules) {
        classicRules.style.display = 'none';
        extendedRules.style.display = 'block';
    }
    
    // Show all gesture guides
    if (lizardGuide && spockGuide) {
        lizardGuide.style.display = 'block';
        spockGuide.style.display = 'block';
    }
    
    // Event listener for capture button
    captureBtn.addEventListener('click', function() {
        // Change button text and disable temporarily
        captureBtn.textContent = 'Processing...';
        captureBtn.disabled = true;
        captureBtn.classList.add('animate-pulse');
        
        // Visual countdown
        let count = 3;
        gameResult.textContent = `Capturing in ${count}...`;
        gameResult.parentElement.style.backgroundColor = 'var(--primary-color)';
        
        const countdownInterval = setInterval(() => {
            count--;
            if (count > 0) {
                gameResult.textContent = `Capturing in ${count}...`;
            } else {
                clearInterval(countdownInterval);
                gameResult.textContent = 'Analyzing gesture...';
                
                // Call the capture endpoint with a cache-busting parameter
                fetch('/capture?t=' + new Date().getTime())
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Network response was not ok');
                        }
                        return response.json();
                    })
                    .then(data => {
                        console.log('Capture response:', data);
                        
                        // Make sure we have valid gesture data
                        if (!data || !data.computer_gesture) {
                            data = {
                                user_gesture: 'unknown',
                                computer_gesture: 'rock',
                                result: 'Error: Could not detect gesture. Try again.'
                            };
                        }
                        
                        // If computer gesture is undefined, set a default
                        if (data.computer_gesture === 'unknown') {
                            data.computer_gesture = 'rock';
                        }
                        
                        // Clear previous results first
                        userGesture.textContent = 'Processing...';
                        computerGesture.textContent = 'Processing...';
                        gameResult.textContent = 'Processing...';
                        userGestureIcon.innerHTML = '';
                        computerGestureIcon.innerHTML = '';
                        
                        // Add a small delay to ensure the UI updates properly
                        setTimeout(() => {
                            // Update the UI with the results
                            updateGameUI(data);
                            
                            // Re-enable the button
                            captureBtn.textContent = 'Capture Gesture';
                            captureBtn.disabled = false;
                            captureBtn.classList.remove('animate-pulse');
                        }, 100);
                    })
                    .catch(error => {
                        console.error('Error capturing gesture:', error);
                        // Update UI with error
                        updateGameUI({
                            user_gesture: 'unknown',
                            computer_gesture: 'rock',
                            result: 'Error: Could not capture gesture. Try again.'
                        });
                        
                        // Re-enable the button
                        captureBtn.textContent = 'Capture Gesture';
                        captureBtn.disabled = false;
                        captureBtn.classList.remove('animate-pulse');
                    });
            }
        }, 1000);
    });
    
    // Add keyboard support (spacebar to capture)
    document.addEventListener('keydown', function(event) {
        if (event.code === 'Space' && !captureBtn.disabled) {
            captureBtn.click();
        }
    });
    
    // Add hover effects to gesture examples
    const gestureExamples = document.querySelectorAll('.gesture-example');
    gestureExamples.forEach(example => {
        const gestureName = example.querySelector('.gesture-name').textContent.toLowerCase();
        if (gestureIcons[gestureName]) {
            const iconElement = document.createElement('div');
            iconElement.className = 'gesture-icon-example';
            iconElement.innerHTML = gestureIcons[gestureName];
            example.appendChild(iconElement);
        }
    });
});
