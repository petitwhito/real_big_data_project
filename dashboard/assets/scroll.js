// JavaScript to automatically scroll to the bottom of the TextArea
document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.getElementById('sql-query-output');
    
    if (textarea) {
        const observer = new MutationObserver(function() {
            textarea.scrollTop = textarea.scrollHeight;
        });
        
        observer.observe(textarea, { childList: true, subtree: true, characterData: true });
    }
});
