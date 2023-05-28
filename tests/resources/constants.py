
db_config = {
    'database': 'advisor_tests',
    'username': 'advisor_user',
    'password': 'advisor_pass',
    'port': 5432
}

workload_config = {
    "size": 2,
    "varying_frequencies": True,
    "training_instances": 10,
    "validation_testing": {
        "number_of_workloads": 5,
        "unknown_query_probabilities": [0.0]
    },
    "excluded_query_classes": [],
    "similar_workloads": False
}

profile_application_view = """
SELECT user_application.id,
       user_application.version,
       user_application.previous,
       user_application.submitted_by,
       user_application.submitted_at,
       user_application.status,
       user_application.type,
       user_application.vehicle_id
FROM user_application
WHERE user_application.type = 10;
"""

accepted_profile_application_view = """
SELECT a.submitted_by,
       b.company_id,
       max(a.version) AS version
FROM profile_application a
         JOIN user_application_company_response b ON a.id = b.application_id
WHERE a.status = 15
  AND b.result = 0
GROUP BY a.submitted_by, b.company_id;
"""

vehicle_application_view = """
SELECT user_application.id,
       user_application.version,
       user_application.previous,
       user_application.submitted_by,
       user_application.submitted_at,
       user_application.status,
       user_application.type,
       user_application.vehicle_id
FROM user_application
WHERE user_application.type = 20;
"""

accepted_vehicle_application_view = """
SELECT a.submitted_by,
       a.vehicle_id,
       b.company_id,
       max(a.version) AS version
FROM vehicle_application a
         JOIN user_application_company_response b ON a.id = b.application_id
WHERE a.status = 15
  AND b.result = 0
GROUP BY a.submitted_by, a.vehicle_id, b.company_id;
"""
