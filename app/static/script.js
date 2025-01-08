document.addEventListener('DOMContentLoaded', function () {
    const chatbotSidebar = document.getElementById('chatbotSidebar');
    const chatButton = document.querySelector('.open-sidebar-btn');
    const chatbotBody = document.querySelector('.chatbot-body');
    const textarea = document.getElementById('chatbotInput');
    const sendButton = document.querySelector('.chatbot-footer button');
    const closeButton = document.querySelector('.chatbot-header button');

    function toggleChatbot() {
        if (chatbotSidebar.style.right === '0px') {
            chatbotSidebar.style.right = '-300px';
            chatButton.style.right = '20px';
        } else {
            chatbotSidebar.style.right = '0px';
            chatButton.style.right = '320px';
        }
    }

    function addMessage(sender, content) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        messageDiv.innerHTML = `<p>${content.replace(/\n/g, '<br>')}</p>`;
        chatbotBody.appendChild(messageDiv);
        chatbotBody.scrollTop = chatbotBody.scrollHeight;
        return messageDiv;
    }

    async function sendMessage() {
        const message = textarea.value.trim();
        if (message === '') return;

        // Add the user message to the chat
        addMessage('user', message);

        // Clear the input and adjust its height
        textarea.value = '';
        adjustTextareaHeight();

        let botMessageDiv = null;
        let botMessage = '';

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');

            while (true) {
                const { value, done } = await reader.read();
                if (done) break; // Stop if the stream is done

                // Decode the chunk as a string
                const chunk = decoder.decode(value, { stream: true });

                // Accumulate the chunk to the bot's message
                botMessage += chunk;

                // Update the bot's message div with the current message
                if (!botMessageDiv) {
                    botMessageDiv = addMessage('bot', botMessage);
                } else {
                    botMessageDiv.querySelector('p').innerHTML = botMessage.replace(/\n/g, '<br>');
                }

                // Scroll to the bottom of the chat to show the latest message
                chatbotBody.scrollTop = chatbotBody.scrollHeight;

                // Add a small delay to ensure the UI updates smoothly
                await new Promise((resolve) => setTimeout(resolve, 10));
            }

            // Handle cases where the response stream ends without any data
            if (!botMessageDiv) {
                addMessage('bot', 'Sorry, I encountered an error processing your request.');
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage('bot', 'Sorry, there was an error processing your request.');
        }
    }

    function adjustTextareaHeight() {
        textarea.style.height = 'auto';
        textarea.style.height = `${textarea.scrollHeight}px`;
    }

    textarea.addEventListener('input', adjustTextareaHeight);

    textarea.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    sendButton.addEventListener('click', sendMessage);
    chatButton.addEventListener('click', toggleChatbot);
    closeButton.addEventListener('click', toggleChatbot);

    // Initial adjustment of textarea height
    adjustTextareaHeight();
});
