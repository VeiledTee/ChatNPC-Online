document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('chat-form');
    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('user-input');
    const characterSelect = document.getElementById('character-select');
    const sendButton = document.getElementById('send-button');

    let selectedCharacter = '';
    let selectedOptionIndex = null;

    characterSelect.addEventListener('change', function () {
        selectedCharacter = characterSelect.options[characterSelect.selectedIndex].text;
        if (characterSelect.value !== '') {
            userInput.disabled = false;
            sendButton.disabled = false;
            sendButton.classList.add('active');

            fetch('/upload_background', {
                method: 'POST',
                body: JSON.stringify({ character: selectedCharacter }),
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => {
                if (response.ok) {
                    console.log("Background uploaded successfully");
                } else {
                    console.error("Error uploading background");
                }
            })
            .catch(error => {
                console.error("Network or other error occurred");
            });
        } else {
            userInput.disabled = true;
            sendButton.disabled = true;
            sendButton.classList.remove('active');
        }
    });

    form.addEventListener('submit', function (e) {
        e.preventDefault();

        const userMessage = userInput.value;

        chatbox.innerHTML += `<p><strong>Player:</strong> ${userMessage}</p>`;

        userInput.value = '';

        fetch('/chat', {
            method: 'POST',
            body: JSON.stringify({ user_input: userMessage, character_select: selectedCharacter, selected_option: selectedOptionIndex }),
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
                chatbox.innerHTML += `<p><strong>${selectedCharacter}: </strong>${responseText}</p>`; // Prepend the response phrase
                chatbox.innerHTML += optionsList;
            } else {
                chatbox.innerHTML += `<p><strong>${selectedCharacter}: </strong>${responseText}</p>`;
            }

            chatbox.scrollTop = chatbox.scrollHeight;
        })
        .catch(error => {
            console.error('Error:', error);
        });

        selectedOptionIndex = null; // Reset the selected option after sending the request
    });

    // Event delegation for option button clicks
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
            fetch('/chat', {
                method: 'POST',
                body: JSON.stringify({ user_input: userMessage, character_select: selectedCharacter, selected_option: selectedOptionIndex }),
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
                    chatbox.innerHTML += `<p><strong>${selectedCharacter}:</strong> ${responseText}</p>`;
                }

                chatbox.scrollTop = chatbox.scrollHeight;
            })
            .catch(error => {
                console.error('Error:', error);
            });

            selectedOptionIndex = null; // Reset the selected option after sending the request
        }
    });

    function getDynamicAudioURLAndPlay() {
        fetch(`/get_latest_audio/${selectedCharacter}`)
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
});