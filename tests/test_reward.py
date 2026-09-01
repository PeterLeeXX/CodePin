from src.rewards.file_localization.file_localization import (
    multilevel_localization_f1_reward,
)


def test_multilevel_reward_scores_added_and_edited_symbols():
    instance = {
        "file_changes": [
            {
                "file": "src/example.py",
                "changes": {
                    "added_modules": ["src/example.py:NewClass"],
                    "added_entities": ["src/example.py:NewClass.run"],
                    "edited_modules": [],
                    "edited_entities": [],
                },
            }
        ]
    }
    locations = [
        {
            "file": "src/example.py",
            "class_name": "NewClass",
            "function_name": "run",
        }
    ]

    reward, details = multilevel_localization_f1_reward(
        instance=instance, structured_locations=locations
    )

    assert reward == 3.0
    assert details["entity_reward"] == 1.0


def test_missing_structured_finish_has_zero_reward():
    reward, _ = multilevel_localization_f1_reward(
        instance={"file_changes": []}, structured_locations=None
    )
    assert reward == 0.0
