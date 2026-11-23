---
permalink: /
title: "Kevin Rapolder"
author_profile: true
hide_title: true
redirect_from:
  - /about/
  - /about.html
---

<section class="home-intro">
  <img class="home-intro__photo" src="{{ site.author.avatar | prepend: '/images/' | relative_url }}" alt="Portrait of {{ site.author.name }}">
  <div class="home-intro__text">
  <h1 class="home-intro__title">Hi, I'm Kevin</h1>
  <p class="home-intro__lead">I'm a Computer Science student (B.Sc.) at the University of Stuttgart. On this site I write about topics I dig into during university projects and seminars &ndash; most recently multi-robot motion planning with diffusion models.</p>
  </div>
</section>

<div class="home-latest">
  <div class="home-latest__header">
    <h2 class="home-latest__title">Latest posts</h2>
    <a class="home-latest__all" href="{{ '/year-archive/' | relative_url }}">All posts <span aria-hidden="true">&rarr;</span></a>
  </div>
  <div class="post-list">
    {% for post in site.posts limit:2 %}{% include post-card.html %}{% endfor %}
  </div>
</div>
