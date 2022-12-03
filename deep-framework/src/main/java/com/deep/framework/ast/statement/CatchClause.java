package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import static com.deep.framework.ast.lexer.TokenType.CATCH;

@Data
public class CatchClause extends Statement {
    private Expression parameter;
    private Statement body;

    public CatchClause(Node prarent, Expression parameter, Statement body) {
        super(prarent);

        this.parameter = parameter;
        this.body = body;

        this.parameter.setPrarent(this);
        this.body.setPrarent(this);

        getChildrens().addAll(parameter, body);
    }

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m.equals(CATCH) && n instanceof ParametersExpression) {
                    //create CatchClause and set Prarent，Parameters
                    CatchClause statement = new CatchClause(node, (Expression) n, (BlockStatement) b);
                    //remove CatchNode and Parameters
                    node.replaceAndRemove(a, statement, b);
                    list.remove(b);
                }
            });
        });
    }
}