document.addEventListener('DOMContentLoaded', () => {

    const data = [
        {
            "category_icon": "fas fa-code",
            "category_name": "Programming Languages",
            "skills": [
                {"name": "Python", "level": "Proficient","logo": "static/images/python.png"},
                {"name": "SQL", "level": "Proficient", "logo": "static/images/sql.png"},
                {"name": "C/C++", "level": "Intermediate", "logo": "static/images/c.png"},
                {"name": "CSS", "level": "Intermediate", "logo": "static/images/css.png"},
                {"name": "HTML", "level": "Intermediate", "logo": "static/images/html.png"},
                {"name": "JavaScript/TypeScript", "level": "Beginner", "logo": "static/images/ts.png"}
            ]
        },
        {
            "category_icon": "fas fa-cogs",
            "category_name": "Machine Learning and AI",
            "skills": [
                {"name": "TensorFlow", "logo": "static/images/tf.png"},
                {"name": "PyTorch", "logo": "static/images/pytorch.png"},
                {"name": "Keras", "logo": "static/images/keras.png"},
                {"name": "Scikit-learn", "logo": "static/images/scikit.png"},
                {"name": "OpenCV", "logo": "static/images/opencv.png"},
                {"name": "Hugging Face Transformers", "logo": "static/images/huggingface.png"}
            ]
        },
        {
            "category_icon": "fas fa-coins",
            "category_name": "Databases and Data Engineering ",
            "skills": [
                {"name": "MySQL", "logo": "static/images/mysql.png"},
                {"name": "MongoDB", "logo": "static/images/mongodb.png"},
                {"name": "Pandas", "logo": "static/images/pandas.png"},
                {"name": "Numpy", "logo": "static/images/numpy.png"},
                {"name": "Spark", "logo": "static/images/spark.png"},
                {"name": "Data modelling", "logo": "static/images/modelling.png"}
            ]
        },
        {
            "category_icon": "fas fa-chart-line",
            "category_name": "BI and Data Visualization",
            "skills": [
                {"name": "Matplotlib", "logo": "static/images/matplot.png"},
                {"name": "Seaborn", "logo": "static/images/seaborn.png"},
                {"name": "Power BI", "logo": "static/images/powerbi.png"},
                {"name": "Streamlit", "logo": "static/images/streamlit.png"}
            ]
        },
        {
            "category_icon": "fas fa-cloud-arrow-up",
            "category_name": "Cloud Computing",
            "skills": [
                {"name": "AWS", "logo": "static/images/aws.png"},
                {"name": "Google Cloud", "logo": "static/images/gcp.png"},
                {"name": "Docker", "logo": "static/images/docker.png"}
            ]
        },
        {
            "category_icon": "fas fa-check-to-slot",
            "category_name": "QA Automation and Testing",
            "skills": [
                {"name": "Selenium", "logo": "static/images/selenium.png"},
                {"name": "Playwright", "logo": "static/images/playwright.png"},
                {"name": "Robot Framework", "logo": "static/images/robot.png"},
                {"name": "Postman", "logo": "static/images/postman.png"},
                {"name": "Swagger", "logo": "static/images/swagger.png"}
            ]
        },
        {
            "category_icon": "fas fa-microchip",
            "category_name": "Electronic Engineering",
            "skills": [
                {"name": "Embedded Software Development", "logo": "static/images/embedded.png"},
                {"name": "Power Electronics", "logo": "static/images/power.png"},
                {"name": "Circuit Design", "logo": "static/images/circuit.png"}
            ]
        },
        {
            "category_icon": "fas fa-users",
            "category_name": "Collaboration and Project Management",
            "skills": [
                {"name": "Agile Methodologies", "logo": "static/images/scrum.png"},
                {"name": "GIT", "logo": "static/images/git.png"},
                {"name": "JIRA", "logo": "static/images/jira.png"},
                {"name": "Notion", "logo": "static/images/notion.png"},
                {"name": "CI/CD", "logo": "static/images/jenkins.png"}
            ]
        },
        {
            "category_icon": "fas fa-language",
            "category_name": "Languages",
            "skills": [
                {"name": "English"},
                {"name": "Spanish"}
            ]
        }
    ];

    const mainContainer = document.getElementById('skills-container');

    const gridContainer = document.createElement('div');
    gridContainer.className = "skills-grid"

    // --- Outer loop for each CATEGORY ---
    data.forEach(category => {
        // 1. Create the main container for the category
        const categoryDiv = document.createElement('div');
        categoryDiv.className = 'skill-category';

        // 2. Create the title (h2)
        const title = document.createElement('h2');

        // 3. Create the icon (i) and set its classes
        const icon = document.createElement('i');
        icon.className = category.category_icon;

        // 4. Add the icon and the title text to the h2 element
        title.appendChild(icon);
        title.append(` ${category.category_name}`); 

        // 5. Create the container for the skill tags
        const skillsContainer = document.createElement('div');
        skillsContainer.className = 'skill-tags';

        // --- Inner loop for each SKILL in the category ---
        category.skills.forEach(skill => {
            const skillTag = document.createElement('div');
            skillTag.className = 'skill-tag';

            if (skill.logo) {
                const logoImg = document.createElement('img');
                logoImg.src = skill.logo;
                logoImg.className = 'skill-logo';
                skillTag.appendChild(logoImg);
            }

            const skillName = document.createElement('span');
            skillName.textContent = skill.name;
            skillTag.appendChild(skillName);

            // Add the skill tag to its container
            skillsContainer.appendChild(skillTag);
        });

        // 6. Append the title and skills container to the main category div
        categoryDiv.appendChild(title);
        categoryDiv.appendChild(skillsContainer);

        // 7. Finally, add the entire category block to the page
        gridContainer.appendChild(categoryDiv);
        mainContainer.appendChild(gridContainer);
    });

});