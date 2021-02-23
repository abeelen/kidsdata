========
Database
========

Usage
-----

For default configuration, simply call :

    $ kidsdata-migrate upgrade heads

This will create an sqlite database in the current directory.

-----------

If you want to use another database file, you can override the DB_DIR variable, either with a regular environment variable or in a .env file in the current directory:

Content of the .env file :

    DB_DIR=/database/path


If you want to use another database engine, override the DB_URI variable. For example with the shared postgresql database on cmaster:

    DB_URI=postgresql://db_user:\*******@cmaster.lam.fr:5432/pipeline

Design
------

Introduction
************
The kidsdata database stores metadata on the files processed by the library. Kidsdata uses sqlalchemy as ORM to query the database and the `Declarative style mapping <https://docs.sqlalchemy.org/en/13/orm/mapping_styles.html#declarative-mapping>`_ to describe the database tables and their columns. `Alembic <https://alembic.sqlalchemy.org/en/latest/>`_ can use these models to (auto)generate and apply migrations on the database.

Kidsdata database should support both postgresql for cluster usage and sqlite for manual/acquisition usage.

The kidsdata database stores mostly metadata on both inputs (scans) and outputs (products) of the data pipeline, and the relations between them.

There is multiple types of scans :
 - Astro :
 - Manual :
 - Tablebt :

These scans have a lot in common but some differences. To represent this situation in the database, we are using the `Joined table inheritance <https://docs.sqlalchemy.org/en/13/orm/inheritance.html#joined-table-inheritance>`_ pattern.

The table `scan` contains all the common columns, and each other table (astro, manual, tablebt) contains the columns specific to each type, as well as a foreign key on the table `scan`. This foreign key is their primary key.

When inserting an astro scan, sqlalchemy automatically inserts a row in the `scan` table, and a row in the table astro, and set the foreign key on the row in astro to the created row in the table scan.

When querying an astro scan, sqlalchemy automatically joins the tables astro and scan on the foreign key and returns an Astro object containing both scan and astro columns.

When querying on the table `scan`, sqlalchemy returns either an Astro, a Manual or a Tablebt object.

The same was done for the different types of products.


Compatibility with outflow
--------------------------

To keep compatibility with the outflow database features (migration, session access in pipeline code, database uri from config file, etc), the inheritance of the classes needed for joined table inheritance is a little bit modified to used mixins instead.

This way, columns are described in simple classes (not inheriting sqlalchemy model base class), and both kidsdata and the outflow pipeline has their own model classes. These models are mixins inheriting both the class describing the columns and the `Base` sqlalchemy model.

The only drawback is that if you mixin sqlalchemy models, you have to change the ForeignKey and Relationship columns to `@declared_attr` (see `Mixing in Relationships <https://docs.sqlalchemy.org/en/14/orm/declarative_mixins.html#mixing-in-relationships>`_)