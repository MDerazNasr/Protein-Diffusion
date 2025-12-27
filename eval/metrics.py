"""Evaluation metrics for protein structures."""

import torch

def masked_mean(x, mask, eps=1e-8):
    '''
    x: (...), mask: same leading shape, bool    
    '''
    mask_f = mask.float()
    return (x * mask_f).sum() / (mask_f.sum().clamp(min=1.0) + eps)

def ca_neighbor_dist(ca_coords, mask):
    """
    Compute CA(i)->CA(i+1) distances.

    ca_coords: (B, L, 3)
    mask:      (B, L) bool

    Returns:
        d: (B, L-1) distances
        d_mask: (B, L-1) valid edges
    """
    diffs = ca_coords[:, 1:, :] - ca_coords[:, :-1, :] #calculate differences between consecutive CAs
    d = torch.sqrt((diffs ** 2).sum(dim=-1) + 1e-8) #calculate euclidean distance
    d_mask = mask[:, 1:] & mask[:, :-1] #calculate valid edges
    d = d * d_mask #apply mask to distances
    return d, d_mask

def pairewise_distances(ca_coords, mask):
    """
    Full pairwise distances between CA atoms (mask-aware).

    ca_coords: (B, L, 3)
    mask:      (B, L) bool

    Returns:
        dist: (B, L, L)
        pair_mask: (B, L, L) True where both residues valid
    """   
    diff = ca_coords[:, :, None, :] - ca_coords[:, None, :, :]
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
    pair_mask = mask[:, :, None] & mask[:, None, :]
    dist = dist * pair_mask
    return dist, pair_mask

def clash_mask(ca_coords, mask, ignore_k=3, clash_dist=0.6):
    """
    Clash score for non-neighbor residues.
    
    Think of a protein like a necklace of beads (amino acids). Each bead has a CA atom.
    A "clash" is when two beads that AREN'T next to each other get too close - like 
    two people bumping into each other when they shouldn't be touching!
    
    Biology context:
    - In real proteins, atoms have a minimum safe distance (like personal space)
    - Neighbors (residues i and i+1) are SUPPOSED to be close (~2 Angstroms apart)
    - But non-neighbors should stay far apart (at least 0.6 Angstroms)
    - If non-neighbors get too close, the protein structure is physically impossible
    
    Args:
        ca_coords: (B, L, 3) CA atom coordinates
        mask: (B, L) bool - which residues are real (not padding)
        ignore_k: int - ignore pairs within k positions in sequence (default 3)
                   Why? Because residues 1,2,3 are naturally close in 3D space
                   even if they're far in the sequence!
        clash_dist: float - distance threshold for a clash (default 0.6 Angstroms)
    
    Returns:
        clash_fraction: (B,) - what fraction of non-neighbor pairs are too close
        min_non_neighbor_dist: (B,) - the smallest distance between any non-neighbor pair
    """
    # Step 1: Calculate distances between ALL pairs of CA atoms
    # This creates a big table: dist[i,j] = distance between residue i and residue j
    dist, pair_mask = pairewise_distances(ca_coords, mask)  # (B, L, L)
    B, L, _ = dist.shape

    # Step 2: Build a "non-neighbor mask" - which pairs should we check for clashes?
    # We only care about pairs that are far apart in the sequence (|i-j| > ignore_k)
    # Example: if ignore_k=3, we ignore pairs like (1,2), (1,3), (1,4) but check (1,5), (1,6), etc.
    idx = torch.arange(L, device=ca_coords.device)
    sep = (idx[:, None] - idx[None, :]).abs()  # |i - j| for all pairs
    non_neighbor = sep > ignore_k  # True if residues are far apart in sequence

    # Step 3: Combine masks - we only check pairs that are:
    #   - Both real residues (pair_mask)
    #   - Far apart in sequence (non_neighbor)
    valid = pair_mask & non_neighbor[None, :, :]  # (B, L, L)

    # Step 4: Find the minimum distance between any non-neighbor pair
    # Set invalid pairs to a huge number so they don't affect the minimum
    big = torch.full_like(dist, 1e9)
    dist_valid = torch.where(valid, dist, big)
    min_non_neighbor_dist = dist_valid.min(dim=-1)[0].min(dim=-1)[0]  # (B,)

    # Step 5: Count clashes - how many non-neighbor pairs are too close?
    clashes = (dist < clash_dist) & valid  # True where distance < threshold AND valid

    # Step 6: Calculate the fraction
    # clash_fraction = (number of clashes) / (total valid non-neighbor pairs)
    valid_pairs = valid.sum(dim=(-1, -2)).clamp(min=1).float()  # (B,)
    clash_count = clashes.sum(dim=(-1, -2)).float()  # (B,)
    clash_fraction = clash_count / valid_pairs  # (B,)

    return clash_fraction, min_non_neighbor_dist

def radius_of_gyration(ca_coords, mask):
    """
    Calculate the radius of gyration (Rg) for protein structures.
    
    Think of a protein like a ball of yarn. The radius of gyration tells you:
    "How spread out is this ball?" or "How compact is this protein?"
    
    Imagine you have a bunch of beads (CA atoms) connected by a string. If you 
    throw them in the air, they'll spread out. The radius of gyration measures 
    the "average distance" each bead is from the center of the whole structure.
    
    Biology context:
    - Compact proteins (like globular proteins) have SMALL Rg - all atoms are 
      close to the center, like a tight ball
    - Extended proteins (like fibrous proteins) have LARGE Rg - atoms spread 
      out far from center, like a stretched string
    - Real proteins typically have Rg values between 5-30 Angstroms depending 
      on size and shape
    - If Rg is too small, the protein is unrealistically compact (like a black hole!)
    - If Rg is too large, the protein is too spread out (like a loose string)
    
    The formula:
    1. Find the center (average position of all CA atoms)
    2. For each CA atom, calculate how far it is from the center
    3. Square those distances
    4. Take the average of all squared distances
    5. Take the square root = radius of gyration!
    
    It's like asking: "If I had to describe this protein as a sphere, 
    what would the radius be?"
    
    Args:
        ca_coords: (B, L, 3) CA atom coordinates
        mask: (B, L) bool - which residues are real (not padding)
    
    Returns:
        rg: (B,) radius of gyration for each protein in the batch
    """
    # Step 1: Find the center of mass (center of the protein)
    # This is just the average position of all CA atoms
    # Think of it like finding the center of a ball - add up all positions and divide by count
    m = mask.unsqueeze(-1).float()  # (B, L, 1) - convert bool to float for math
    denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1, 1) - count of real residues
    mean = (ca_coords * m).sum(dim=1, keepdim=True) / denom  # (B, 1, 3) - center point
    
    # Step 2: Calculate how far each CA atom is from the center
    # For each atom, find: distance = sqrt((x - center_x)² + (y - center_y)² + (z - center_z)²)
    centered = (ca_coords - mean) * m  # (B, L, 3) - distance vector from center to each atom
    sq = (centered ** 2).sum(dim=-1)  # (B, L) - squared distance from center for each atom
    
    # Step 3: Average the squared distances (but only for real residues, ignore padding)
    # This gives us the "mean squared distance from center"
    mean_squared_dist = sq.sum(dim=1) / mask.sum(dim=1).clamp(min=1).float()  # (B,)
    
    # Step 4: Take the square root to get the radius of gyration
    # This is the final answer: "On average, how far are atoms from the center?"
    rg = torch.sqrt(mean_squared_dist + 1e-8)  # (B,) - add tiny number to avoid sqrt(0)
    
    return rg
def compute_backbone_metrics(ca_coords, mask):
    """
    Main one-call metrics function - your "health check" for protein structures!
    
    Think of this like a doctor's checkup for a protein. Just like a doctor measures
    your height, weight, blood pressure, and heart rate, this function measures different
    "vital signs" of a protein structure to see if it's healthy and realistic.
    
    What does it check?
    This function measures 6 important things about your protein:
    
    1. **CA-CA mean distance**: How far apart are neighboring amino acids on average?
       - Like measuring the average step size when walking
       - Should be around 2.0 Angstroms (healthy range: 1.9-2.1)
       - Too small = atoms squished together (unrealistic!)
       - Too large = atoms too spread out (protein falling apart!)
    
    2. **CA-CA min distance**: What's the shortest distance between neighbors?
       - Like finding the smallest step in your walk
       - Should be > 1.2 Angstroms
       - If too small, atoms are overlapping (physically impossible!)
    
    3. **CA-CA max distance**: What's the longest distance between neighbors?
       - Like finding the biggest step in your walk
       - Should be < 4.5 Angstroms
       - If too large, the protein chain is broken or stretched too far!
    
    4. **Clash fraction**: What percentage of non-neighbor atoms are too close?
       - Like checking how many people are bumping into each other in a crowd
       - Should be 0% (no clashes!)
       - If > 0%, the structure has atoms overlapping (bad!)
    
    5. **Min non-neighbor distance**: What's the smallest distance between any two
       atoms that AREN'T neighbors?
       - Like finding the closest two people who shouldn't be near each other
       - Should be > 0.6 Angstroms
       - If too small, atoms are crashing into each other!
    
    6. **Radius of gyration**: How compact or spread out is the protein?
       - Like measuring if a ball of yarn is tight or loose
       - Typical range: 5-30 Angstroms depending on size
       - Too small = unrealistically compact (like a black hole!)
       - Too large = too spread out (like a loose string!)
    
    Why do we need all these metrics?
    - Real proteins have specific geometric constraints (atoms can't overlap!)
    - Generated proteins might look okay but have hidden problems
    - These metrics catch problems before you try to use the protein for anything
    
    Args:
        ca_coords: (B, L, 3) CA atom coordinates for a batch of proteins
        mask: (B, L) bool - which residues are real (not padding)
    
    Returns:
        A dictionary with 6 metrics (all averaged across the batch):
        {
            "ca_ca_mean": average neighbor distance,
            "ca_ca_min": minimum neighbor distance,
            "ca_ca_max": maximum neighbor distance,
            "clash_fraction": fraction of non-neighbor pairs that clash,
            "min_non_neighbor_dist": smallest non-neighbor distance,
            "radius_of_gyration": how compact the protein is
        }
    """
    # Part 1: Measure neighbor distances (consecutive amino acids)
    # This is like measuring the distance between each bead and the next bead on a necklace
    d, dmask = ca_neighbor_dist(ca_coords, mask)  # (B, L-1) distances between neighbors
    
    # Calculate statistics: mean, min, and max of neighbor distances
    # We only count real bonds (ignore padding)
    d_mean = masked_mean(d, dmask).item()  # Average step size
    d_min = (torch.where(dmask, d, torch.full_like(d, 1e9))).min().item()  # Smallest step
    d_max = (torch.where(dmask, d, torch.full_like(d, -1e9))).max().item()  # Largest step

    # Part 2: Check for clashes (atoms that are too close together)
    # This is like checking if any beads that aren't next to each other are touching
    clash_frac, min_non_nb = clash_mask(ca_coords, mask, clash_dist=2.5, ignore_k=3)
    clash_frac_mean = clash_frac.mean().item()  # What % of pairs are too close?
    min_non_nb_mean = min_non_nb.mean().item()  # What's the smallest distance found?

    # Part 3: Measure compactness (how spread out is the protein?)
    # This is like measuring if the necklace is bunched up or stretched out
    rg = radius_of_gyration(ca_coords, mask)  # (B,) - compactness for each protein
    rg_mean = rg.mean().item()  # Average compactness across batch

    # Return all the "vital signs" in one dictionary
    return {
        "ca_ca_mean": d_mean,              # Average neighbor distance
        "ca_ca_min": d_min,                # Shortest neighbor distance
        "ca_ca_max": d_max,                # Longest neighbor distance
        "clash_fraction": clash_frac_mean,  # % of non-neighbors that are too close
        "min_non_neighbor_dist": min_non_nb_mean,  # Smallest non-neighbor distance
        "radius_of_gyration": rg_mean,     # How compact/spread out
    }