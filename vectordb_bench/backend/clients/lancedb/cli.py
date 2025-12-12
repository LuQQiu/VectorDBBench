from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB
from ..api import IndexType


class LanceDBTypedDict(CommonTypedDict):
    uri: Annotated[
        str,
        click.option("--uri", type=str, help="URI connection string", required=True),
    ]
    api_key: Annotated[
        str | None,
        click.option("--api-key", type=str, help="API key for authentication", required=False),
    ]
    host_override: Annotated[
        str | None,
        click.option("--host-override", type=str, help="Host override for LanceDB connection", required=False),
    ]
    table_name: Annotated[
        str | None,
        click.option(
            "--table-name",
            type=str,
            default=None,
            help="Table name (if not provided, uses lancedb_bench_test with random suffix)",
        ),
    ]

@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBTypedDict)
def LanceDB(**parameters: Unpack[LanceDBTypedDict]):
    from .config import LanceDBConfig, _lancedb_case_config

    run(
        db=DB.LanceDB,
        db_config=LanceDBConfig(
            db_label=parameters["db_label"],
            uri=parameters["uri"],
            api_key=SecretStr(parameters["api_key"]) if parameters.get("api_key") else None,
            host_override=parameters.get("host_override"),
            table_name=parameters.get("table_name"),
        ),
        db_case_config=_lancedb_case_config.get("NONE")(),
        **parameters,
    )


class LanceDBAutoIndexTypedDict(CommonTypedDict, LanceDBTypedDict):
    nprobes: Annotated[
        int,
        click.option(
            "--nprobes", type=int, default=20, help="Number of IVF partitions to search (higher = better recall, slower)"
        ),
    ]
    refine_factor: Annotated[
        int,
        click.option(
            "--refine-factor", type=int, default=3, help="Re-rank top k*refine_factor results with exact distance"
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBAutoIndexTypedDict)
def LanceDBAutoIndex(**parameters: Unpack[LanceDBAutoIndexTypedDict]):
    from .config import LanceDBAutoIndexConfig, LanceDBConfig

    run(
        db=DB.LanceDB,
        db_config=LanceDBConfig(
            db_label=parameters["db_label"],
            uri=parameters["uri"],
            api_key=SecretStr(parameters["api_key"]) if parameters.get("api_key") else None,
            host_override=parameters.get("host_override"),
            table_name=parameters.get("table_name"),
        ),
        db_case_config=LanceDBAutoIndexConfig(
            nprobes=parameters["nprobes"],
            refine_factor=parameters["refine_factor"],
        ),
        **parameters,
    )


class LanceDBIVFPQTypedDict(CommonTypedDict, LanceDBTypedDict):
    num_partitions: Annotated[
        int,
        click.option(
            "--num-partitions",
            type=int,
            default=0,
            help="Number of partitions for IVFPQ index, unset = use LanceDB default",
        ),
    ]
    num_sub_vectors: Annotated[
        int,
        click.option(
            "--num-sub-vectors",
            type=int,
            default=0,
            help="Number of sub-vectors for IVFPQ index, unset = use LanceDB default",
        ),
    ]
    nbits: Annotated[
        int,
        click.option(
            "--nbits",
            type=int,
            default=8,
            help="Number of bits for IVFPQ index (must be 4 or 8), unset = use LanceDB default",
        ),
    ]
    nprobes: Annotated[
        int,
        click.option(
            "--nprobes", type=int, default=0, help="Number of probes for IVFPQ search, unset = use LanceDB default"
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBIVFPQTypedDict)
def LanceDBIVFPQ(**parameters: Unpack[LanceDBIVFPQTypedDict]):
    from .config import LanceDBConfig, LanceDBIndexConfig

    run(
        db=DB.LanceDB,
        db_config=LanceDBConfig(
            db_label=parameters["db_label"],
            uri=parameters["uri"],
            api_key=SecretStr(parameters["api_key"]) if parameters.get("api_key") else None,
            host_override=parameters.get("host_override"),
            table_name=parameters.get("table_name"),
        ),
        db_case_config=LanceDBIndexConfig(
            index=IndexType.IVFPQ,
            num_partitions=parameters["num_partitions"],
            num_sub_vectors=parameters["num_sub_vectors"],
            nbits=parameters["nbits"],
            nprobes=parameters["nprobes"],
        ),
        **parameters,
    )


class LanceDBHNSWTypedDict(CommonTypedDict, LanceDBTypedDict):
    m: Annotated[int, click.option("--m", type=int, default=0, help="HNSW parameter m")]
    ef_construction: Annotated[
        int, click.option("--ef-construction", type=int, default=0, help="HNSW parameter ef_construction")
    ]
    ef: Annotated[int, click.option("--ef", type=int, default=0, help="HNSW search parameter ef")]


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceDBHNSWTypedDict)
def LanceDBHNSW(**parameters: Unpack[LanceDBHNSWTypedDict]):
    from .config import LanceDBConfig, LanceDBHNSWIndexConfig

    run(
        db=DB.LanceDB,
        db_config=LanceDBConfig(
            db_label=parameters["db_label"],
            uri=parameters["uri"],
            api_key=SecretStr(parameters["api_key"]) if parameters.get("api_key") else None,
            host_override=parameters.get("host_override"),
            table_name=parameters.get("table_name"),
        ),
        db_case_config=LanceDBHNSWIndexConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef=parameters["ef"],
        ),
        **parameters,
    )
