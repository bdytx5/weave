from ..api import op
from .. import weave_types as types
from . import wb_domain_types as wdt
from .wandb_domain_gql import (
    gql_prop_op,
    gql_direct_edge_op,
)

import urllib


# Section 1/6: Tag Getters
# None

# Section 2/6: Root Ops
# None

# Section 3/6: Attribute Getters
# artifact_membership_version_index is written in the plain style
# because the attribute is part of the required fragment
@op(name="artifactMembership-versionIndex")
def artifact_membership_version_index(
    artifactMembership: wdt.ArtifactCollectionMembership,
) -> int:
    return artifactMembership.gql["versionIndex"]


gql_prop_op(
    "artifactMembership-createdAt",
    wdt.ArtifactCollectionMembershipType,
    "createdAt",
    types.Timestamp(),
)

# Section 4/6: Direct Relationship Ops
gql_direct_edge_op(
    "artifactMembership-collection",
    wdt.ArtifactCollectionMembershipType,
    "artifactCollection",
    wdt.ArtifactCollectionType,
)

gql_direct_edge_op(
    "artifactMembership-artifactVersion",
    wdt.ArtifactCollectionMembershipType,
    "artifact",
    wdt.ArtifactVersionType,
)

gql_direct_edge_op(
    "artifactMembership-aliases",
    wdt.ArtifactCollectionMembershipType,
    "aliases",
    wdt.ArtifactAliasType,
    is_many=True,
)

# Section 5/6: Connection Ops
# None

# Section 6/6: Non Standard Business Logic Ops
# None


@op(name="artifactMembership-link")
def artifact_membership_link(
    artifactMembership: wdt.ArtifactCollectionMembership,
) -> wdt.Link:
    return wdt.Link(
        name=f"{artifactMembership.gql['artifactCollection']['name']}:v{artifactMembership.gql['versionIndex']}",
        url=f"/{artifactMembership.gql['artifactCollection']['defaultArtifactType']['project']['entity']['name']}/"
        f"{artifactMembership.gql['artifactCollection']['defaultArtifactType']['project']['name']}/"
        f"artifacts/{urllib.parse.quote(artifactMembership.gql['artifactCollection']['defaultArtifactType']['name'])}/"
        f"{urllib.parse.quote(artifactMembership.gql['artifactCollection']['name'])}"
        f"/v{artifactMembership.gql['versionIndex']}",
    )
