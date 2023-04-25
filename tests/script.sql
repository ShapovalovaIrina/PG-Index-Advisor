create table if not exists profiles
(
    user_id                bigserial
        primary key,
    username               varchar(255),
    deleted                boolean default false not null,
    contacts               jsonb
);

create unique index if not exists profiles_username_index
    on profiles (username);

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

create unique index if not exists user_application_previous_index
    on user_application (previous);

create unique index if not exists user_application_submitted_by_version_index
    on user_application (submitted_by, version)
    where (type = 10);

create unique index if not exists user_application_vehicle_id_version_index
    on user_application (vehicle_id, version)
    where (type = 20);

