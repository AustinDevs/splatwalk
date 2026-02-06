document.addEventListener('DOMContentLoaded', function() {
  const jobId = window.JOB_ID;
  const jobStatus = window.JOB_STATUS;

  if (jobStatus === 'completed' || jobStatus === 'failed') {
    return;
  }

  const statusLabels = {
    pending: 'Queued',
    scraping: 'Scraping Zillow',
    uploading: 'Uploading Images',
    provisioning: 'Starting GPU',
    processing_viewcrafter: 'Running ViewCrafter',
    processing_instantsplat: 'Running InstantSplat',
    processing_combined: 'Running Combined Pipeline',
    converting: 'Converting to KSPLAT',
    completed: 'Complete',
    failed: 'Failed',
  };

  const statusDescriptions = {
    pending: 'Waiting for an available processing slot...',
    scraping: 'Extracting photos and metadata from the Zillow listing...',
    uploading: 'Uploading images to cloud storage...',
    provisioning: 'Spinning up GPU server for 3D reconstruction...',
    processing_viewcrafter: 'Generating novel views with ViewCrafter AI...',
    processing_instantsplat: 'Creating Gaussian splat with InstantSplat...',
    processing_combined: 'Running combined MASt3R pipeline...',
    converting: 'Converting PLY to optimized KSPLAT format...',
    completed: 'Your VR tour is ready!',
    failed: 'Something went wrong during processing.',
  };

  const statusOrder = [
    'pending', 'scraping', 'uploading', 'provisioning',
    'processing_viewcrafter', 'processing_instantsplat',
    'processing_combined', 'converting', 'completed'
  ];

  const statusLabel = document.getElementById('status-label');
  const statusDescription = document.getElementById('status-description');
  const progressFill = document.getElementById('progress-fill');
  const progressValue = document.getElementById('progress-value');

  // Connect to SSE
  const eventSource = new EventSource('/api/jobs/' + jobId + '/events');

  eventSource.onmessage = function(event) {
    try {
      const data = JSON.parse(event.data);

      if (data.status) {
        updateStatus(data.status);
      }

      if (data.progress !== undefined) {
        updateProgress(data.progress);
      }

      if (data.status === 'completed') {
        eventSource.close();
        window.location.href = '/view/' + jobId;
      }

      if (data.status === 'failed') {
        eventSource.close();
        window.location.reload();
      }
    } catch (err) {
      console.error('Failed to parse SSE event:', err);
    }
  };

  eventSource.onerror = function() {
    console.log('SSE connection error, will retry...');
  };

  function updateStatus(status) {
    if (statusLabel) {
      statusLabel.textContent = statusLabels[status] || status;
    }
    if (statusDescription) {
      statusDescription.textContent = statusDescriptions[status] || '';
    }

    // Update timeline
    const currentIndex = statusOrder.indexOf(status);
    const timelineItems = document.querySelectorAll('.timeline-item');

    timelineItems.forEach(function(item, index) {
      item.classList.remove('complete', 'current');

      if (index < currentIndex) {
        item.classList.add('complete');
        const marker = item.querySelector('.timeline-marker');
        if (marker) marker.textContent = '✓';
      } else if (index === currentIndex && status !== 'completed' && status !== 'failed') {
        item.classList.add('current');
      }

      if (item.dataset.status === 'completed' && status === 'completed') {
        item.classList.add('complete');
        const marker = item.querySelector('.timeline-marker');
        if (marker) marker.textContent = '✓';
      }
    });
  }

  function updateProgress(progress) {
    if (progressFill) {
      progressFill.style.width = progress + '%';
    }
    if (progressValue) {
      progressValue.textContent = Math.round(progress);
    }
  }
});
