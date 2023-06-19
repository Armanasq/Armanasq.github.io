---
# Leave the homepage title empty to use the site title
title:
date: 2022-10-24
type: landing

sections:
  - block: about.biography
    id: about
    content:
      title: Biography
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: admin
  - block: collection
    content:
      title: Recent Publications
      text: |-
        {{% callout note %}}
        Quickly discover relevant content by [filtering publications](./publication/).
        {{% /callout %}}
      filters:
        folders:
          - publication
        exclude_featured: true
    design:
      columns: '2'
      view: citation

  - block: accomplishments
    content:
      title: Academic Seervices
      # Date format for experience
      #   Refer to https://wowchemy.com/docs/customization/#date-format
      date_format: Jan 2006
      # Experiences.
      #   Add/remove as many `experience` items below as you like.
      #   Required fields are `title`, `company`, and `date_start`.
      #   Leave `date_end` empty if it's your current employer.
      #   Begin multi-line descriptions with YAML's `|2-` multi-line prefix.
      items:
        - title: Referee of Research Council
          company: Students’ Scientific Research Center
          date_start: '2019-04-01'
          date_end: ''
          description: |2-
              Responsibilities include:

              * Analyzed and evaluated research proposals to determine if they are appropriate for funding
        - title: Review for Conferences
          date_start: '2023-01-01'
          date_end: ''
          description: |2-

            * International Federation of Automatic Control (IFAC) World Congress 2023, Yokohama, Japan+
        - title: Review for Journals
          date_start: '2023-01-02'
          date_end: ''
          description: |2-
              Including

            * IEEE Transactions on Instrumentation & Measurement , 6 Papers
            * The Aeronautical Journal , 3 Papers
            * Aerospace Science and Technology , 3 Papers
              * Complete list on [www.webofscience.com/wos/author/record/AEM-0212-2022](https://www.webofscience.com/wos/author/record/AEM-0212-2022)
    design:
      columns: '2'
  - block: markdown
    content:
      title: Teaching Experience
      text: | 
            | |
            | ------------------------------- | -------------------------------- | ----------------------- |
            | Teaching Assistant              | University of Tehran  </br> * Course: Fuzzy Logic (Graduate Level) </br> * Instructor: Dr. M.H. Sabour             | Fall 2022               |
            | Instructor                      | Aviation Industry Training Center </br> * 11 Courses taught on Electronics, Navigation, Aviation (Undergraduate)  | Sep. 2019 - Sep. 2021  |
            | Thesis Supervisor               | Aviation Industry Training Center </br> Supervised Theses </br> * Design and Implementation of a 3 Axis CNC Machine () </br> * Design and Implementation of Pulse Circuits Training Board () </br> * Design, Simulation, and Building of an Aircraft Fire Extinguishing System () </br> * Design and Implementation of Retractable Landing Gear () </br> * Design and Implementation of a CNC Hot Wire () |  </br>  </br> Spring 2021 - Fall 2021 </br> Fall 2020 - Fall 2021 </br> Spring 2020 - Fall 2020 </br> Fall 2019 - Spring 2020 </br> Fall 2019 - Spring 2020 |
      filters:
        folders:
          - post
    design:
      # Choose how many columns the section has. Valid values: '1' or '2'.
      columns: '2'
      # Choose your content listing view - here we use the `showcase` view
      view: showcase
      # For the Showcase view, do you want to flip alternate rows?
      flip_alt_rows: true
  - block: collection
    id: posts
    content:
      title: Recent Posts
      subtitle: ''
      text: ''
      # Choose how many pages you would like to display (0 = all pages)
      count: 5
      # Filter on criteria
      filters:
        folders:
          - post
        author: ""
        category: ""
        tag: ""
        exclude_featured: false
        exclude_future: false
        exclude_past: false
        publication_type: ""
      # Choose how many pages you would like to offset by
      offset: 0
      # Page order: descending (desc) or ascending (asc) date.
      order: desc
    design:
      # Choose a layout view
      view: compact
      columns: '2'
  - block: portfolio
    id: projects
    content:
      title: Projects
      filters:
        folders:
          - project
      # Default filter index (e.g. 0 corresponds to the first `filter_button` instance below).
      default_button_index: 0
      # Filter toolbar (optional).
      # Add or remove as many filters (`filter_button` instances) as you like.
      # To show all items, set `tag` to "*".
      # To filter by a specific tag, set `tag` to an existing tag name.
      # To remove the toolbar, delete the entire `filter_button` block.
      buttons:
        - name: All
          tag: '*'
        - name: Deep Learning
          tag: Deep Learning
    design:
      # Choose how many columns the section has. Valid values: '1' or '2'.
      columns: '1'
      view: showcase
      # For Showcase view, flip alternate rows?
      flip_alt_rows: false
  - block: markdown
    content:
      title: Gallery
      subtitle: ''
      text: |-
        {{% gallery album="demo" %}}
    design:
      columns: '1'
  
  - block: collection
    id: talks
    content:
      title: Recent Talks
      filters:
        folders:
          - event
    design:
      columns: '2'
      view: compact
  - block: tag_cloud
    content:
      title: Popular Topics
    design:
      columns: '2'
  - block: contact
    id: contact
    content:
      title: Contact
      subtitle:
      text: |-

# Contact (add or remove contact options as necessary)
      email: a.asgharpoor1993@gmail.com
      phone: +98 919 609 7597
      appointment_url: 'https://calendly.com/arman-asgharpoor/30min'
      address:
        street: Innovation Center, SSRC, Vesal st.
        city: Tehran
        country: Iran
        country_code: IR

      contact_links:
        - icon: skype
          icon_pack: fab
          name: Skype Me
          link: 'skype:armannearu?call'
      # Automatically link email and phone or display as text?
      autolink: true
      # Email form provider
      form:
        provider: netlify
        formspree:
          id:
        netlify:
          # Enable CAPTCHA challenge to reduce spam?
          captcha: true
    design:
      columns: '2'
---
