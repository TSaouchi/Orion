function loadHtml(url, containerId) {
    var plotContainer = document.getElementById(containerId);
    if (plotContainer) {
        if (!plotContainer.innerHTML) {
            var iframe = document.createElement("iframe");
            iframe.src = url;
            iframe.width = "800";
            iframe.height = "900";
            plotContainer.appendChild(iframe);
            plotContainer.style.display = "block";
        } else {
            // If content is already loaded, remove it
            plotContainer.innerHTML = "";
            plotContainer.style.display = "none";
        }
    }
}

document.addEventListener('DOMContentLoaded', function() {
    var currentVersion = document.querySelector('.rst-current-version');
    var otherVersions = document.querySelector('.rst-other-versions');
    
    if (currentVersion && otherVersions) {
        currentVersion.addEventListener('click', function() {
            if (otherVersions.style.display === 'none' || !otherVersions.style.display) {
                otherVersions.style.display = 'block';
            } else {
                otherVersions.style.display = 'none';
            }
        });
    }
});