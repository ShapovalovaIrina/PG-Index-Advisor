create table partsupp
(
    ps_partkey    numeric(18),
    ps_suppkey    integer,
    ps_availqty   integer,
    ps_supplycost double precision,
    ps_comment    varchar(199)
);

alter table partsupp
    owner to postgres;

create table lineitem
(
    l_orderkey      numeric(18),
    l_partkey       numeric(18),
    l_suppkey       integer,
    l_linenumber    integer,
    l_quantity      double precision,
    l_extendedprice double precision,
    l_discount      double precision,
    l_tax           double precision,
    l_returnflag    char,
    l_linestatus    char,
    l_shipdate      timestamp,
    l_commitdate    timestamp,
    l_receiptdate   timestamp,
    l_shipinstruct  char(25),
    l_shipmode      char(10),
    l_comment       varchar(44)
);

alter table lineitem
    owner to postgres;

create table supplier
(
    s_suppkey   integer,
    s_name      char(25),
    s_address   varchar(40),
    s_nationkey integer,
    s_phone     char(15),
    s_acctbal   double precision,
    s_comment   varchar(101)
);

alter table supplier
    owner to postgres;

create table customer
(
    c_custkey    numeric(18),
    c_name       varchar(25),
    c_address    varchar(40),
    c_nationkey  integer,
    c_phone      char(15),
    c_acctbal    double precision,
    c_mktsegment char(10),
    c_comment    varchar(117)
);

alter table customer
    owner to postgres;

create table part
(
    p_partkey     numeric(18),
    p_name        varchar(55),
    p_mfgr        char(25),
    p_brand       char(10),
    p_type        varchar(25),
    p_size        integer,
    p_container   char(10),
    p_retailprice double precision,
    p_comment     varchar(23)
);

alter table part
    owner to postgres;

create table public.orders
(
    o_orderkey      numeric(18),
    o_custkey       numeric(18),
    o_orderstatus   char,
    o_totalprice    double precision,
    o_orderdate     timestamp,
    o_orderpriority char(15),
    o_clerk         char(15),
    o_shippriority  integer,
    o_comment       varchar(79)
);

alter table public.orders
    owner to postgres;

create table region
(
    r_regionkey integer,
    r_name      char(25),
    r_comment   varchar(152)
);

alter table region
    owner to postgres;

create table nation
(
    n_nationkey integer,
    n_name      char(25),
    n_regionkey integer,
    n_comment   varchar(152)
);

alter table nation
    owner to postgres;


