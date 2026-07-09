# Paste sample dataset builder here (generate_sample_data.py)
"""
Utility script to generate sample online courses dataset and its configuration.
Saves outputs in data/ directory so that app.py can run instantly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def generate_courses():
    courses_data = [
        {
            "course_title": "JavaScript Intermediate and Advanced Concepts",
            "category": "Web Development",
            "level": "Intermediate",
            "language": "English",
            "instructor": "M. Osei",
            "price": 49.99,
            "enrollments": 18000,
            "avg_rating": 4.9,
            "reviews_count": 1400,
            "completion_pct": 62.0,
            "wishlist_count": 2800,
            "instructor_score": 4.85,
            "course_age_months": 14,
        },
        {
            "course_title": "Advanced React Patterns",
            "category": "Web Development",
            "level": "Advanced",
            "language": "English",
            "instructor": "L. Fernandez",
            "price": 79.99,
            "enrollments": 9000,
            "avg_rating": 4.8,
            "reviews_count": 700,
            "completion_pct": 55.0,
            "wishlist_count": 1500,
            "instructor_score": 4.7,
            "course_age_months": 8,
        },
        {
            "course_title": "Python for Everybody",
            "category": "Data Science",
            "level": "Beginner",
            "language": "English",
            "instructor": "R. Nakamura",
            "price": 0.0,
            "enrollments": 250000,
            "avg_rating": 4.7,
            "reviews_count": 50000,
            "completion_pct": 25.0,
            "wishlist_count": 30000,
            "instructor_score": 4.6,
            "course_age_months": 48,
        },
        {
            "course_title": "Complete Web Developer Bootcamp",
            "category": "Web Development",
            "level": "Beginner",
            "language": "English",
            "instructor": "D. Whitfield",
            "price": 199.99,
            "enrollments": 420000,
            "avg_rating": 4.6,
            "reviews_count": 85000,
            "completion_pct": 18.0,
            "wishlist_count": 40000,
            "instructor_score": 4.5,
            "course_age_months": 60,
        },
        {
            "course_title": "Intro to HTML and CSS",
            "category": "Web Development",
            "level": "Beginner",
            "language": "Spanish",
            "instructor": "D. Whitfield",
            "price": 19.99,
            "enrollments": 500000,
            "avg_rating": 4.3,
            "reviews_count": 90000,
            "completion_pct": 15.0,
            "wishlist_count": 60000,
            "instructor_score": 4.2,
            "course_age_months": 72,
        },
        {
            "course_title": "Data Analysis with Pandas",
            "category": "Data Science",
            "level": "Intermediate",
            "language": "English",
            "instructor": "R. Nakamura",
            "price": 49.99,
            "enrollments": 45000,
            "avg_rating": 4.6,
            "reviews_count": 5600,
            "completion_pct": 34.0,
            "wishlist_count": 6100,
            "instructor_score": 4.5,
            "course_age_months": 22,
        },
    ]

    config = {
        "label": "Online Course Catalog",
        "identifier_column": "course_title",
        "channels": [
            {
                "column": "avg_rating",
                "label": "Average rating",
                "color": "teal",
                "normalize": "minmax",
                "normalize_scope": "global",
                "default_transform": {"type": "amplify", "power": 4.0, "scale": 2.5},
            },
            {
                "column": "completion_pct",
                "label": "Completion rate",
                "color": "lime",
                "normalize": "minmax",
                "normalize_scope": "global",
                "default_transform": {"type": "amplify", "power": 2.0, "scale": 2.2},
            },
            {
                "column": "enrollments",
                "label": "Enrollments",
                "color": "amber",
                "normalize": "log_minmax",
                "normalize_scope": "global",
                "default_transform": {"type": "linear", "w": 0.6},
                "warning": "Highly skewed distribution — log scaling recommended.",
            },
            {
                "column": "reviews_count",
                "label": "Reviews count",
                "color": "purple",
                "normalize": "log_minmax",
                "normalize_scope": "global",
                "default_transform": {"type": "linear", "w": 0.4},
            },
            {
                "column": "wishlist_count",
                "label": "Wishlist count",
                "color": "pink",
                "normalize": "log_minmax",
                "normalize_scope": "global",
                "default_transform": {"type": "linear", "w": 0.4},
            },
            {
                "column": "course_age_months",
                "label": "Course age",
                "color": "sky",
                "normalize": "minmax",
                "normalize_scope": "global",
                "default_transform": {"type": "clamp", "max_v": 0.7, "scale": 0.8},
                "warning": "Mild recency bonus, capped so it doesn't dominate.",
            },
            {
                "column": "instructor_score",
                "label": "Instructor score",
                "color": "rose",
                "normalize": "minmax",
                "normalize_scope": "global",
                "default_transform": {"type": "linear", "w": 1.0},
            },
        ],
        "filters": [
            {"column": "category", "label": "Category", "type": "categorical"},
            {"column": "level", "label": "Level", "type": "categorical"},
            {"column": "language", "label": "Language", "type": "categorical"},
            {"column": "price", "label": "Price ($)", "type": "range"},
        ],
    }

    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    df = pd.DataFrame(courses_data)
    df.to_pickle(data_dir / "sample_courses.pkl")
    (data_dir / "sample_courses.config.json").write_text(json.dumps(config, indent=2))

    print(f"Generated data/sample_courses.pkl with {len(df)} records.")
    print("Generated data/sample_courses.config.json configuration mapping.")


if __name__ == "__main__":
    generate_courses()
