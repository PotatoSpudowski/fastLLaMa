const ghpages = require('gh-pages');

ghpages.publish('dist', (err) => {
    if (err) {
        console.error('Failed to deploy:', err);
    } else {
        console.log('Deployment successful');
    }
});
