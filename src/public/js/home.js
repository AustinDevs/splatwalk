document.addEventListener('DOMContentLoaded', function() {
  const form = document.getElementById('submit-form');
  const urlInput = document.getElementById('listing-url');
  const submitBtn = document.getElementById('submit-btn');
  const errorMessage = document.getElementById('error-message');

  form.addEventListener('submit', async function(e) {
    e.preventDefault();

    const url = urlInput.value.trim();

    if (!url) {
      showError('Please enter a Realtor.com URL');
      return;
    }

    if (!url.includes('realtor.com')) {
      showError('Please enter a valid Realtor.com URL');
      return;
    }

    try {
      submitBtn.disabled = true;
      submitBtn.textContent = 'Creating...';
      hideError();

      const response = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ listingUrl: url }),
      });

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to create job');
      }

      window.location.href = '/jobs/' + data.data.id;
    } catch (err) {
      showError(err.message);
      submitBtn.disabled = false;
      submitBtn.textContent = 'Create VR Tour';
    }
  });

  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
  }

  function hideError() {
    errorMessage.style.display = 'none';
  }
});
