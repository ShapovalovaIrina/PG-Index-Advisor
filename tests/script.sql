-- PROFILES TABLE

create table if not exists profiles
(
    user_id                bigserial
        primary key,
    username               varchar(255),
    deleted                boolean default false not null,
    contacts               jsonb
);

-- PROFILES INDEXES

create unique index if not exists profiles_username_index
    on profiles (username);

-- USER_APPLICATION TABLE

create table if not exists user_application
(
    id                              bigserial
        primary key,
    version                         integer                  not null,
    previous                        bigint,
    submitted_by                    bigint                   not null
        references profiles,
    submitted_at                    timestamp with time zone,
    status                          integer                  not null,
    type                            integer                  not null,
    vehicle_id                      varchar(255)
);

-- USER_APPLICATION INDEXES

create unique index if not exists user_application_previous_index
    on user_application (previous);

create unique index if not exists user_application_submitted_by_version_index
    on user_application (submitted_by, version)
    where (type = 10);

create unique index if not exists user_application_vehicle_id_version_index
    on user_application (vehicle_id, version)
    where (type = 20);

-- APPLICATION RESPONSES TABLE

create table if not exists user_application_response
(
    id             bigserial
        primary key,
    application_id bigint                   not null
        references user_application,
    company_id     bigint,
    reviewer_type  integer                  not null,
    result         integer                  not null
    constraint review_origin
        check ((reviewer_type = 10) OR ((reviewer_type = 20) AND (company_id IS NOT NULL)))
);

-- COMPANY RESPONSES INDEX

create unique index if not exists user_application_response_application_id_reviewer_type_company_
    on user_application_response (application_id, reviewer_type, company_id);

-- COMPANY RESPONSES VIEW

create or replace view user_application_company_response
            (id, application_id, company_id, reviewer_type, result) as
SELECT user_application_response.id,
       user_application_response.application_id,
       user_application_response.company_id,
       user_application_response.reviewer_type,
       user_application_response.result
FROM user_application_response
WHERE user_application_response.reviewer_type = 20;

-- PROFILE APPLICATION VIEWS

create or replace view profile_application
            (id, version, previous, submitted_by, submitted_at, status,
             type, vehicle_id)
as
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

create or replace view accepted_profile_application(submitted_by, company_id, version) as
SELECT a.submitted_by,
       b.company_id,
       max(a.version) AS version
FROM profile_application a
         JOIN user_application_company_response b ON a.id = b.application_id
WHERE a.status = 15
  AND b.result = 0
GROUP BY a.submitted_by, b.company_id;

-- VEHICLE APPLICATION VIEWS

create or replace view vehicle_application
            (id, version, previous, submitted_by, submitted_at, status,
             type, vehicle_id)
as
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

create or replace view accepted_vehicle_application(submitted_by, vehicle_id, company_id, version) as
SELECT a.submitted_by,
       a.vehicle_id,
       b.company_id,
       max(a.version) AS version
FROM vehicle_application a
         JOIN user_application_company_response b ON a.id = b.application_id
WHERE a.status = 15
  AND b.result = 0
GROUP BY a.submitted_by, a.vehicle_id, b.company_id;


