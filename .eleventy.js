module.exports = function (eleventyConfig) {
  eleventyConfig.addPassthroughCopy({
    "src/css": "css",
    "src/javascript": "javascript",
    "src/assets": "assets",
  });

  return {
    dir: {
      input: "src",
      includes: "_includes",
      output: "_site",
    },
    htmlTemplateEngine: "njk",
    markdownTemplateEngine: "njk",
  };
};
