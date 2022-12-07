package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.NodeList;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.*;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.type.Type;
import lombok.Data;

import java.util.List;
import java.util.Objects;

@Data
public class VariableDeclaration extends Declaration {
    private Type type;
    private Expression name;
    private Expression initializer;
    private static Type variableType;

    public VariableDeclaration(Node prarent, Type type, Name name) {
        super(prarent);
        this.type = type;
        this.name = name;

        this.type.setPrarent(this);
        this.name.setPrarent(this);

        getChildrens().addAll(type, name);
    }

    public VariableDeclaration(Node prarent, Type type, Expression name, Expression initializer) {
        super(prarent);
        this.type = type;
        this.name = name;
        this.initializer = initializer;

        this.type.setPrarent(this);
        this.name.setPrarent(this);
        this.initializer.setPrarent(this);

        getChildrens().addAll(type, name, initializer);
    }

    public static void parser(Node node) {
        if (node.getPrarent() instanceof ParametersExpression) return;
        if (node instanceof MethodDeclaration) return;
        if (node instanceof FieldDeclaration) return;
        if (node instanceof VariableDeclaration) return;
        List<Node> nodes = node.getChildrens().stream().filter(a -> Field_Modifiers.contains(a.getTokenType())).toList();
        node.getChildrens().removeAll(nodes);
        variableType = null;
        Stream.of(node.getChildrens()).reduce((list, a, b, c) -> {
            if (a instanceof Name && b instanceof Name) {
                VariableDeclaration declare = new VariableDeclaration(node, variableType = Type.getType(a), (Name) b);
                node.replaceAndRemove(a, declare, b);
                list.remove(b);
            } else if (a instanceof Type && b instanceof Name) {
                VariableDeclaration declare = new VariableDeclaration(node, variableType = (Type) a, (Name) b);
                node.replaceAndRemove(a, declare, b);
                list.remove(b);
            } else if (a instanceof Name && b instanceof AssignExpression d) {
                VariableDeclaration declare = new VariableDeclaration(node, variableType = Type.getType(a), d.getVariable(), d.getValue());
                d.getValue().setPrarent(declare);
                node.replaceAndRemove(a, declare, b);
                list.remove(b);
            } else if (a instanceof Type && b instanceof AssignExpression d) {
                VariableDeclaration declare = new VariableDeclaration(node, variableType = (Type) a, d.getVariable(), d.getValue());
                d.getValue().setPrarent(declare);
                node.replaceAndRemove(a, declare, b);
                list.remove(b);
            } else if (Objects.nonNull(variableType) && a.equals(TokenType.COMMA) && b instanceof Name) {
                VariableDeclaration declare = new VariableDeclaration(node, variableType, (Name) a);
                node.replaceAndRemove(a, declare, b);
                list.remove(b);
            } else if (Objects.nonNull(variableType) && a.equals(TokenType.COMMA) && b instanceof AssignExpression e) {
                VariableDeclaration declare = new VariableDeclaration(node, variableType, e.getVariable(), e.getValue());
                node.replaceAndRemove(a, declare, b);
                list.remove(b);
            }
        });

        if (Objects.nonNull(variableType) && node.getChildrens().size() > 1) {
            NodeList<VariableDeclaration> variableDeclarations = new NodeList(node.getChildrens());
            VariableExpression variableExpression = new VariableExpression(node.getPrarent(), variableDeclarations);
            node.getPrarent().replace(node, variableExpression);
        }
    }

    @Override
    public String toString() {
        String concat = type.toString().concat(" ").concat(name.toString()).concat(" ");
        if (Objects.nonNull(initializer)) concat = concat.concat("= ").concat(initializer.toString()).concat(";");
        return concat;
    }
}