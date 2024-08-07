// Wait for the DOM to be fully loaded before executing the script
window.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('floatingImages'); // Get the container element for floating images
    const imageNames = ['emotion_0012.jpg', 'emotion_0031.jpg', 'emotion_0090.jpg', 'emotion_0157.jpg', 'emotion_0268.jpg']; // List of image filenames

    // Loop to create and append 50 floating images
    for (let i = 0; i < 50; i++) {
        const img = document.createElement('img'); // Create a new img element
        img.src = `/static/images/${imageNames[i % imageNames.length]}`; // Set the source of the image
        img.className = 'floating-img'; // Assign a class name to the image
        img.style.position = 'absolute'; // Set the image position to absolute
        img.style.top = `${Math.random() * 100}%`; // Randomize the top position
        img.style.left = `${Math.random() * 100}%`; // Randomize the left position
        img.style.transform = `translate(-50%, -50%) scale(${Math.random() * 0.5 + 3})`; // Apply random scale transformation
        img.style.opacity = 0; // Set initial opacity to 0

        container.appendChild(img); // Append the image to the container

        // Fade in the image with a slight delay based on its index
        setTimeout(() => img.style.opacity = 0.6, 10 + 100 * i);

        // Randomize the speed of movement in both X and Y directions
        const speedX = Math.random() * 0.1 - 0.05;
        const speedY = Math.random() * 0.1 - 0.05;

        // Move the image in random directions at regular intervals
        setInterval(() => {
            const topVal = parseFloat(img.style.top); // Get the current top position
            const leftVal = parseFloat(img.style.left); // Get the current left position
            img.style.top = `${(topVal + speedY + 100) % 100}%`; // Update the top position
            img.style.left = `${(leftVal + speedX + 100) % 100}%`; // Update the left position
        }, 100); // Interval of 100 milliseconds
    }
});

// Function to check the status of a file
function checkStatus(filename) {
    // Make an AJAX request to check the status of the file
    $.getJSON('/check_status/' + filename, function(data) {
        if (data.ready) {
            // If the file is ready, redirect to the recordings page
            window.location.href = '/recordings/' + filename;
        } else {
            // If the file is not ready, check again after 5 seconds
            setTimeout(() => checkStatus(filename), 5000);
        }
    }).fail(function() {
        // Log an error message if the AJAX request fails
        console.log("Error checking status");
        // Check again after 5 seconds
        setTimeout(() => checkStatus(filename), 5000);
    });
}
