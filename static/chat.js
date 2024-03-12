document.addEventListener('DOMContentLoaded', function () {
    let selectedCharacter = '';
    let selectedOptionIndex = null;
    let username = ''; // Define username variable in the outer scope

    const usernameForm = document.getElementById('username-form');
    const mainContent = document.getElementById('main-content');
    const popupContainer = document.getElementById('popup-container');

    // Hide the popup initially
    popupContainer.style.display = 'none';

    // Listen for form submission
    usernameForm.addEventListener('submit', function (event) {
        event.preventDefault(); // Prevent the default form submission behavior

        // Retrieve the username entered by the user
        const usernameInput = document.getElementById('username');
        username = usernameInput.value; // Update the outer scoped username variable
        console.log("Username:", username);

        // Hide the username container
        const usernameContainer = document.getElementById('username-container');
        usernameContainer.style.display = 'none';

        // Show the main content
        mainContent.style.display = 'flex';

        // Scroll to the main content
        mainContent.scrollIntoView({ behavior: 'smooth' });

        // Show the popup after scrolling finishes
        setTimeout(showPopup, 1000); // Adjust the delay as needed
    });

    // Event listener for the close popup button
    const closePopupBtn = document.getElementById('close-popup-btn');
    closePopupBtn.addEventListener('click', function () {
        hidePopup();
    });

    // Function to show the popup
    function showPopup() {
        popupContainer.style.display = 'block';
    }

    // Function to hide the popup
    function hidePopup() {
        popupContainer.style.display = 'none';
    }

    // Listen for form submission (this one seems redundant, you may consider removing it)
    usernameForm.addEventListener('submit', function (event) {
        event.preventDefault(); // Prevent the default form submission behavior

        // Hide the username container
        const usernameContainer = document.getElementById('username-container');
        usernameContainer.style.display = 'none';

        // Show the main content
        mainContent.style.display = 'flex';

        // Scroll to the main content
        mainContent.scrollIntoView({ behavior: 'smooth' });

        // Show the popup after scrolling finishes
        setTimeout(showPopup, 1000); // Adjust the delay as needed
    });

    const characterList = document.getElementById('character-list');

    characterList.addEventListener('click', function (e) {
        if (e.target && e.target.tagName === 'LI') {
            selectedCharacter = e.target.dataset.character; // Assign selectedCharacter here
            console.log("Selected character:", selectedCharacter);
            console.log("Selected character:", selectedCharacter);
            // Make an AJAX request to a Flask endpoint with the selected character's name
            fetch('/select-character', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ character: selectedCharacter })
            })
            .then(response => {
                if (response.ok) {
                    console.log("Character selected successfully.");
                    // Hide the popup when a character is selected
                    hidePopup();
                    // Upload background after character is selected
                    fetch('/upload_background', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ character: selectedCharacter })
                    })
                    .then(response => {
                        if (response.ok) {
                            console.log("Background uploaded successfully for", selectedCharacter);
                        } else {
                            console.error("Failed to upload background.");
                        }
                    })
                    .catch(error => {
                        console.error("Error uploading background:", error);
                    });

                } else {
                    console.error("Failed to select character.");
                }
            })
            .catch(error => {
                console.error("Error selecting character:", error);
            });

            // Once a character is selected, enable input and send button
            const userInput = document.getElementById('user-input');
            const sendButton = document.getElementById('send-button');
            userInput.disabled = false;
            sendButton.disabled = false;
        }
    });

    const form = document.getElementById('chat-form');
    const chatbox = document.getElementById('chatbox');
        const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    // Function to populate the input box with placeholder text
    function showPlaceholderText() {
        userInput.value = 'Type your message...';
    }

    // Add event listener to sidebar buttons
    const sidebarButtons = document.querySelectorAll('.sidebar li');
    sidebarButtons.forEach(button => {
        button.addEventListener('click', function () {
            // Your existing button click logic...

            // Show placeholder text in the input box
            showPlaceholderText();
        });
    });

    // Additional logic to handle placeholder text when the input box is focused
    userInput.addEventListener('focus', function () {
        // Clear placeholder text when input box is focused
        if (userInput.value === 'Type your message...') {
            userInput.value = '';
        }
    });

    userInput.addEventListener('blur', function () {
        // Restore placeholder text if input box is left empty
        if (userInput.value === '') {
            showPlaceholderText();
        }
    });

    form.addEventListener('submit', function (e) {
        e.preventDefault();

        const userMessage = userInput.value;

        chatbox.innerHTML += `<p><strong>${username}:</strong> ${userMessage}</p>`;
        userInput.value = '';
        console.log("fetch 1 pre chat character", selectedCharacter)
        // Ensure selectedCharacter is not null before sending the /chat request
        if (selectedCharacter) {
            fetch('/chat', {
                method: 'POST',
                body: JSON.stringify({ user_input: userMessage, character: selectedCharacter, selected_option: selectedOptionIndex }),
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                const responseText = data.response;
                const options = data.options;

                if (options && options.length > 0) {
                    const optionsList = options.map((option, index) =>
                        `<div class="options-container">
                            <button class="option-btn ${selectedOptionIndex === index ? 'selected' : 'blurred'}"
                                    data-index="${index}" ${selectedOptionIndex !== null ? 'disabled' : ''}>
                                ${option}
                            </button>
                        </div>`
                    ).join('');
                    chatbox.innerHTML += `<p><strong>${nameConversion(toSnakeCase=false, toConvert=selectedCharacter)}: </strong>${responseText}</p>`; // Prepend the response phrase
                    chatbox.innerHTML += optionsList;
            } else {
                chatbox.innerHTML += `<p><strong>${nameConversion(toSnakeCase=false, toConvert=selectedCharacter)}: </strong>${responseText}</p>`;
            }

            chatbox.scrollTop = chatbox.scrollHeight;
        })
        .catch(error => {
            console.error('Error:', error);
        });

        selectedOptionIndex = null; // Reset the selected option after sending the request
        } else {
            console.error('No character selected.'); // Log an error if selectedCharacter is null
        }
    });

    chatbox.addEventListener('click', function (event) {
        const target = event.target;
        if (target.classList.contains('option-btn') && !target.classList.contains('selected') && selectedOptionIndex === null) {
            selectedOptionIndex = parseInt(target.getAttribute('data-index'));

            // Remove 'selected' class from previously selected button
            const selectedButton = document.querySelector('.option-btn.selected');
            if (selectedButton) {
                selectedButton.classList.remove('selected');
            }

            target.classList.add('selected');

            // Change background color of selected button
            target.style.backgroundColor = '#1A85FF';

            // Blur the remaining unselected options
            const optionButtons = document.querySelectorAll('.option-btn');
            optionButtons.forEach((button, index) => {
                if (index !== selectedOptionIndex) {
                    button.classList.add('blurred');
                    button.disabled = true; // Disable the remaining unselected options
                }
            });

            // After selecting the option, send the request
            const userMessage = '';  // You can modify this to include the actual user's message
            console.log("fetch 2 pre chat character", selectedCharacter)
            fetch('/chat', {
                method: 'POST',
                body: JSON.stringify({ user_input: userMessage, character: selectedCharacter, selected_option: selectedOptionIndex }),
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                const responseText = data.response;
                const options = data.options;

                if (options && options.length > 0) {
                    const optionsList = options.map((option, index) =>
                        `<div class="options-container">
                            <button class="option-btn ${selectedOptionIndex === index ? 'selected' : 'blurred'}"
                                    data-index="${index}" ${selectedOptionIndex !== null ? 'disabled' : ''}>
                                ${option}
                            </button>
                        </div>`
                    ).join('');
                    chatbox.innerHTML += optionsList;
                } else {
                    chatbox.innerHTML += `<p><strong>${nameConversion(toSnakeCase=false, toConvert=selectedCharacter)}:</strong> ${responseText}</p>`;
                }

                chatbox.scrollTop = chatbox.scrollHeight;
            })
            .catch(error => {
                console.error('Error:', error);
            });

            selectedOptionIndex = null; // Reset the selected option after sending the request
        }
    });

    document.getElementById('run-script-button').addEventListener('click', function() {
        fetch('/clean-slate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            if (response.ok) {
                return response.json();
            }
            throw new Error('Failed to execute script');
        })
        .then(data => {
            console.log(data.message); // Log success message
        })
        .catch(error => {
            console.error('Error:', error); // Log any errors
        });
    });

    characterList.addEventListener('click', function (e) {
        if (e.target && e.target.tagName === 'LI') {
            // Toggle 'clicked' class
            e.target.classList.toggle('clicked');

            // Your existing code for selecting a character
        }
    });

    function getDynamicAudioURLAndPlay() {
        fetch(`/get_latest_audio/${nameConversion(toSnakeCase=false, toConvert=selectedCharacter)}`)
        .then(response => response.json())
        .then(data => {
            const latestAudioURL = data.latest_audio_url;
            console.log(latestAudioURL);

            if (latestAudioURL) {
                const autoPlayAudio = document.getElementById('auto-play-audio');
                autoPlayAudio.querySelector('source').src = latestAudioURL;
                autoPlayAudio.load();
                autoPlayAudio.play();
            } else {
                console.error("No audio files found for the selected character");
            }
        })
        .catch(error => {
            console.error("Error fetching the latest audio URL:", error);
        });
    }

    function nameConversion(toSnakeCase, toConvert) {
        if (toSnakeCase) {
            return toConvert.toLowerCase().split(" ").join("_");
        } else {
            return toConvert.split("_").map(word => {
                return word.charAt(0).toUpperCase() + word.slice(1);
            }).join(" ").replace(/(-)\s*([a-zA-Z])/g, (match, p1, p2) => {
                return p1 + p2.toUpperCase();
            });
        }
    }

});