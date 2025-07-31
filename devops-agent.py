from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
import boto3
import os

# Set up LLM with Groq
llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model="mixtral-8x7b-32768"
)

# Get EC2 instance status using boto3
def get_ec2_status():
    ec2 = boto3.client('ec2', region_name='us-east-1')
    response = ec2.describe_instance_status(IncludeAllInstances=True)
    statuses = response.get("InstanceStatuses", [])
    if not statuses:
        return "No EC2 instances are currently running."
    return "\n".join([
        f"Instance {s['InstanceId']} is {s['InstanceState']['Name']}."
        for s in statuses
    ])

# CrewAI Agent setup
agent = Agent(
    role="AWS DevOps Agent",
    goal="Check and report EC2 instance status",
    backstory="You are an expert in AWS infrastructure and automation.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Create Task
task = Task(
    description="Check status of EC2 instances in us-east-1 and report in plain English.",
    expected_output="A clear summary of all EC2 instance states.",
    agent=agent
)

# Crew runs this task
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True
)

if __name__ == "__main__":
    ec2_summary = get_ec2_status()
    print("🔍 AWS EC2 Status:\n", ec2_summary)
    print("🤖 LLM Interpretation:")
    crew.kickoff(inputs={"context": ec2_summary})
