// Main JavaScript for Rock Paper Scissors Lizard Spock Game

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const captureBtn = document.getElementById('captureBtn');
    const userGesture = document.getElementById('userGesture');
    const computerGesture = document.getElementById('computerGesture');
    const gameResult = document.getElementById('gameResult');
    const userGestureIcon = document.getElementById('userGestureIcon');
    const computerGestureIcon = document.getElementById('computerGestureIcon');
    const pyTorchBtn = document.getElementById('pyTorchBtn');
    const openCVBtn = document.getElementById('openCVBtn');
    const processingText = document.getElementById('processingText');
    
    // Processing option buttons
    const grayscaleBtn = document.getElementById('grayscaleBtn');
    const binaryBtn = document.getElementById('binaryBtn');
    const adaptiveBtn = document.getElementById('adaptiveBtn');
    const edgeDetectionBtn = document.getElementById('edgeDetectionBtn');
    const skinDetectionBtn = document.getElementById('skinDetectionBtn');
    const blurBtn = document.getElementById('blurBtn');
    const contourBtn = document.getElementById('contourBtn');
    const cannyBtn = document.getElementById('cannyBtn');
    
    // Sliders
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdValue = document.getElementById('thresholdValue');
    const blurSlider = document.getElementById('blurSlider');
    const blurValue = document.getElementById('blurValue');
    
    // Game and processing state
    let detectionMethod = 'pytorch'; // 'pytorch' or 'opencv'
    let processingMethod = 'skin'; // 'grayscale', 'binary', 'adaptive', 'edge', 'skin', 'blur', 'contour', 'canny'
    let thresholdVal = 127;
    let blurVal = 5;
    
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
    
    // Function to update the active detection method
    function updateDetectionMethod(method) {
        detectionMethod = method;
        
        // Update UI
        if (method === 'pytorch') {
            pyTorchBtn.classList.add('active');
            openCVBtn.classList.remove('active');
        } else {
            pyTorchBtn.classList.remove('active');
            openCVBtn.classList.add('active');
        }
        
        // Send method preference to server
        fetch('/set_detection_method', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ method: method })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Detection method updated:', data);
            // Refresh video feeds to show the change
            refreshVideoFeeds();
        })
        .catch(error => {
            console.error('Error updating detection method:', error);
        });
    }
    
    // Function to update the active processing method
    function updateProcessingMethod(method) {
        processingMethod = method;
        
        // Update UI
        const allButtons = [
            grayscaleBtn, 
            binaryBtn, 
            adaptiveBtn, 
            edgeDetectionBtn, 
            skinDetectionBtn, 
            blurBtn, 
            contourBtn, 
            cannyBtn
        ];
        allButtons.forEach(btn => btn.classList.remove('active'));
        
        // Set active class
        switch(method) {
            case 'grayscale':
                grayscaleBtn.classList.add('active');
                processingText.textContent = 'Grayscale is active';
                break;
            case 'binary':
                binaryBtn.classList.add('active');
                processingText.textContent = 'Binary Threshold is active';
                break;
            case 'adaptive':
                adaptiveBtn.classList.add('active');
                processingText.textContent = 'Adaptive Threshold is active';
                break;
            case 'edge':
                edgeDetectionBtn.classList.add('active');
                processingText.textContent = 'Edge Detection is active';
                break;
            case 'skin':
                skinDetectionBtn.classList.add('active');
                processingText.textContent = 'Skin Detection is active';
                break;
            case 'blur':
                blurBtn.classList.add('active');
                processingText.textContent = 'Blur is active';
                break;
            case 'contour':
                contourBtn.classList.add('active');
                processingText.textContent = 'Contour Detection is active';
                break;
            case 'canny':
                cannyBtn.classList.add('active');
                processingText.textContent = 'Canny Edge Detection is active';
                break;
        }
        
        // Send processing preference to server
        fetch('/set_processing_method', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                method: method,
                threshold: thresholdVal,
                blur: blurVal
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Processing method updated:', data);
            // Refresh video feeds to show the change
            refreshVideoFeeds();
        })
        .catch(error => {
            console.error('Error updating processing method:', error);
        });
    }
    
    // Function to update threshold value
    function updateThreshold(value) {
        thresholdVal = parseInt(value);
        thresholdValue.textContent = value;
        
        // Send to server
        fetch('/set_processing_params', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                threshold: thresholdVal,
                blur: blurVal
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Threshold updated:', data);
            refreshVideoFeeds();
        })
        .catch(error => {
            console.error('Error updating threshold:', error);
        });
    }
    
    // Function to update blur value
    function updateBlur(value) {
        blurVal = parseInt(value);
        blurValue.textContent = value;
        
        // Send to server
        fetch('/set_processing_params', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                threshold: thresholdVal,
                blur: blurVal
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Blur updated:', data);
            refreshVideoFeeds();
        })
        .catch(error => {
            console.error('Error updating blur:', error);
        });
    }
    
    // Function to refresh video feeds to prevent caching
    function refreshVideoFeeds() {
        // Refresh both video feeds by adding a timestamp to force reload
        const timestamp = new Date().getTime();
        
        // Get the video feed elements
        const videoFeed = document.getElementById('video_feed');
        const processedFeed = document.getElementById('processed_feed');
        
        // Update the src attributes with a new timestamp and active parameters
        if (videoFeed) {
            // Create a URL object to manipulate the URL parts safely
            const currentSrc = new URL(videoFeed.src, window.location.origin);
            // Remove existing timestamp parameter if present
            currentSrc.searchParams.delete('t');
            // Add new timestamp
            currentSrc.searchParams.set('t', timestamp);
            // Update the src
            videoFeed.src = currentSrc.toString();
        }
        
        if (processedFeed) {
            // Create a URL object to manipulate the URL parts safely
            const currentSrc = new URL(processedFeed.src, window.location.origin);
            // Clear existing parameters that might cause conflicts
            currentSrc.searchParams.delete('t');
            currentSrc.searchParams.delete('method');
            currentSrc.searchParams.delete('processing');
            currentSrc.searchParams.delete('threshold');
            currentSrc.searchParams.delete('blur');
            // Add all current parameters
            currentSrc.searchParams.set('t', timestamp);
            currentSrc.searchParams.set('method', detectionMethod);
            currentSrc.searchParams.set('processing', processingMethod);
            currentSrc.searchParams.set('threshold', thresholdVal);
            currentSrc.searchParams.set('blur', blurVal);
            // Update the src
            processedFeed.src = currentSrc.toString();
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
            gameResult.parentElement.style.background = 'linear-gradient(to right, var(--success-color), #2ecc71)'; // Green gradient for win
        } else if (result.result.includes('Computer wins')) {
            gameResult.parentElement.style.background = 'linear-gradient(to right, var(--secondary-color), var(--secondary-dark))'; // Red gradient for loss
        } else if (result.result.includes('Tie')) {
            gameResult.parentElement.style.background = 'linear-gradient(to right, var(--warning-color), #e67e22)'; // Orange gradient for tie
        } else {
            // Error or other cases
            gameResult.parentElement.style.background = 'linear-gradient(to right, #7f8c8d, #95a5a6)'; // Gray gradient for error
        }
    }
    
    // Event listeners for detection method toggle
    pyTorchBtn.addEventListener('click', () => {
        updateDetectionMethod('pytorch');
    });
    
    openCVBtn.addEventListener('click', () => {
        updateDetectionMethod('opencv');
    });
    
    // Event listeners for processing method toggle
    grayscaleBtn.addEventListener('click', () => {
        updateProcessingMethod('grayscale');
    });
    
    binaryBtn.addEventListener('click', () => {
        updateProcessingMethod('binary');
    });
    
    adaptiveBtn.addEventListener('click', () => {
        updateProcessingMethod('adaptive');
    });
    
    edgeDetectionBtn.addEventListener('click', () => {
        updateProcessingMethod('edge');
    });
    
    skinDetectionBtn.addEventListener('click', () => {
        updateProcessingMethod('skin');
    });
    
    blurBtn.addEventListener('click', () => {
        updateProcessingMethod('blur');
    });
    
    contourBtn.addEventListener('click', () => {
        updateProcessingMethod('contour');
    });
    
    cannyBtn.addEventListener('click', () => {
        updateProcessingMethod('canny');
    });
    
    // Event listeners for sliders
    thresholdSlider.addEventListener('input', () => {
        updateThreshold(thresholdSlider.value);
    });
    
    blurSlider.addEventListener('input', () => {
        updateBlur(blurSlider.value);
    });
    
    // Event listener for capture button
    captureBtn.addEventListener('click', function() {
        // Change button text and disable temporarily
        captureBtn.textContent = 'Processing...';
        captureBtn.disabled = true;
        captureBtn.classList.add('animate-pulse');
        
        // Visual countdown
        let count = 3;
        gameResult.textContent = `Capturing in ${count}...`;
        gameResult.parentElement.style.background = 'linear-gradient(to right, var(--primary-color), var(--primary-dark))';
        
        const countdownInterval = setInterval(() => {
            count--;
            if (count > 0) {
                gameResult.textContent = `Capturing in ${count}...`;
            } else {
                clearInterval(countdownInterval);
                gameResult.textContent = 'Analyzing gesture...';
                
                // Call the capture endpoint with additional parameters
                fetch(`/capture?t=${new Date().getTime()}&method=${detectionMethod}&processing=${processingMethod}&threshold=${thresholdVal}&blur=${blurVal}`)
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
                            captureBtn.innerHTML = '<i class="fas fa-hand-pointer"></i> Capture Gesture';
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
                        captureBtn.innerHTML = '<i class="fas fa-hand-pointer"></i> Capture Gesture';
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
    
    // Initialize with default settings
    updateDetectionMethod('pytorch');
    updateProcessingMethod('skin');
});
