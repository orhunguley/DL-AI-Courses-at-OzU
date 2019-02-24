package src.search;

import java.util.ArrayList;
import java.util.LinkedList;

public class AI_DFS {

	static int[] initial_state = { 0, 0, 0, 3, 1, 2 };
	static int[] goal_state = { 3, 1, 2, 0, 0, 0 };
	static boolean solutionFound;

	static LinkedList<Candidate> candidateList = new LinkedList<Candidate>();


	static class Candidate {

		public int no;
		public int ExpectedSalary;
		public int PreferredDepartment;
		public boolean hired;

		public Candidate(int ExpectedSalary, int PreferredDepartment, int no) {
			this.no = no;
			this.ExpectedSalary = ExpectedSalary;
			this.PreferredDepartment = PreferredDepartment;

		}

	}

	static class Node {

		public ArrayList<Node> children = new ArrayList<Node>();
		public Candidate candidate = null;
		public Node parent = null;
		public int[] state;
		public boolean hiredOrNot;
		public int totalCost = 0;

		public Node(int[] state) {
			this.state = state;
		}

		public Node(Node parent, Candidate candidate) {
			this.candidate = candidate;
			this.parent = parent;

		}

	}

	public static void makeList() {

		candidateList.add(new Candidate(50, 1, 1));
		candidateList.add(new Candidate(30, 2, 2));
		candidateList.add(new Candidate(85, 1, 3));
		candidateList.add(new Candidate(55, 3, 4));
		candidateList.add(new Candidate(45, 3, 5));
		candidateList.add(new Candidate(30, 3, 6));
		candidateList.add(new Candidate(80, 1, 7));
		candidateList.add(new Candidate(55, 2, 8));
		candidateList.add(new Candidate(30, 1, 9));
		candidateList.add(new Candidate(35, 1, 10));

	}

	public static void createTree(Node node, Candidate candidate, int depth) {

		if (depth == candidateList.size()) {

			if (candidate != null) {
				node.state[candidate.PreferredDepartment - 1] = node.state[candidate.PreferredDepartment - 1] + 1;
				node.state[candidate.PreferredDepartment - 1 + 3] = node.state[candidate.PreferredDepartment - 1 + 3]
						- 1;
				node.totalCost = node.totalCost + candidate.ExpectedSalary;
			}
			return;
		}

		depth = depth + 1;

		if (depth - 1 < candidateList.size())

			if (node.parent != null) {

				if (candidate != null) {

					node.candidate = candidate;
					node.state[candidate.PreferredDepartment - 1] = node.state[candidate.PreferredDepartment - 1] + 1;
					node.state[candidate.PreferredDepartment - 1
							+ 3] = node.state[candidate.PreferredDepartment - 1 + 3] - 1;

					node.totalCost = node.totalCost + candidate.ExpectedSalary;

					node.children.add(new Node(node, candidateList.get(depth - 1)));
					node.children.add(new Node(node, null));

					node.children.get(0).state = node.state.clone();
					node.children.get(1).state = node.state.clone();

					node.children.get(0).totalCost = node.totalCost;
					node.children.get(1).totalCost = node.totalCost;

					createTree(node.children.get(0), candidateList.get(depth - 1), depth);
					createTree(node.children.get(1), null, depth);
				} else {

					node.children.add(new Node(node, candidateList.get(depth - 1)));
					node.children.add(new Node(node, null));

					node.children.get(0).state = node.state.clone();
					node.children.get(1).state = node.state.clone();

					node.children.get(0).totalCost = node.totalCost;
					node.children.get(1).totalCost = node.totalCost;

					createTree(node.children.get(0), candidateList.get(depth - 1), depth);
					createTree(node.children.get(1), null, depth);
				}

			} else {

				node.children.add(new Node(node, candidateList.get(depth - 1)));
				node.children.add(new Node(node, null));
				node.children.get(0).state = node.state.clone();
				node.children.get(1).state = node.state.clone();

				node.children.get(0).totalCost = node.totalCost;
				node.children.get(1).totalCost = node.totalCost;

				createTree(node.children.get(0), candidateList.get(0), depth);
				createTree(node.children.get(1), null, depth);
			}

	}

	public static void DepthFirstSearch(Node root) {

		if (solutionFound == true) {
			return;
		}

		if (root.children.isEmpty()) {

			return;

		} else {

			if (root.state[0]==3 && root.state[1]==1 && root.state[2]==2) {

				System.out.println("Feasible solution found!");
				System.out.println("Hired candidates are: ");
				Node n;
				n = root;

				while (n != null) {

					if (n.candidate != null) {

						System.out.print(n.candidate.no + " ");

					}

					n = n.parent;

				}
				
				System.out.println("");
				System.out.println("Hired candidates are: " + root.totalCost);

				solutionFound = true;
				return;

			}

			DepthFirstSearch(root.children.get(0));

			DepthFirstSearch(root.children.get(1));
			
		}

	}

	public static void main(String[] args) {
		int i = 0;
		Node root = new Node(initial_state);
		root.totalCost = 0;
		makeList();
		createTree(root, null, 0);

		DepthFirstSearch(root);

		i = i + 1;
	}

}
